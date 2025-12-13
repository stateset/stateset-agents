"""
API Dependencies Module

Dependency injection for authentication, authorization, and common services.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .services.auth_service import get_auth_service, AuthService
from .security import get_api_security_monitor
from utils.security import SecurityMonitor


# ============================================================================
# Authentication Types
# ============================================================================

@dataclass
class AuthenticatedUser:
    """Authenticated user information."""
    user_id: str
    roles: List[str]
    email: Optional[str] = None
    name: Optional[str] = None
    metadata: Optional[Dict] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "AuthenticatedUser":
        """Create AuthenticatedUser from dictionary."""
        return cls(
            user_id=data.get("user_id", data.get("sub", "anonymous")),
            roles=data.get("roles", []),
            email=data.get("email"),
            name=data.get("name"),
            metadata={k: v for k, v in data.items() if k not in ("user_id", "sub", "roles", "email", "name")},
        )


# ============================================================================
# Security Setup
# ============================================================================

security = HTTPBearer(auto_error=False)  # Don't auto-error to allow custom handling
security_monitor = SecurityMonitor()


# ============================================================================
# Authentication Dependencies
# ============================================================================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthenticatedUser:
    """
    Get current authenticated user from JWT token.

    Args:
        credentials: HTTP Bearer credentials from request.
        auth_service: Authentication service for token validation.

    Returns:
        AuthenticatedUser with user information.

    Raises:
        HTTPException 401: If authentication fails.
    """
    api_security = get_api_security_monitor()

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        user_data = auth_service.authenticate_token(token)
        if not user_data:
            # Log auth failure
            is_locked, remaining = api_security.log_auth_failure(
                client_ip="unknown",
                path="auth",
                reason="Invalid token",
            )
            if is_locked:
                raise HTTPException(
                    status_code=403,
                    detail="Account temporarily locked due to too many failed attempts",
                )
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Clear any previous failures on successful auth
        api_security.auth_tracker.clear_failures("unknown")

        return AuthenticatedUser.from_dict(user_data)

    except HTTPException:
        raise
    except Exception as e:
        security_monitor.log_security_event(
            "authentication_failure",
            {"error": str(e), "token_prefix": token[:10] if token else "None"},
        )
        raise HTTPException(
            status_code=401,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    auth_service: AuthService = Depends(get_auth_service),
) -> Optional[AuthenticatedUser]:
    """
    Get current user if authenticated, None otherwise.

    Useful for endpoints that work both authenticated and anonymously.
    """
    if not credentials:
        return None

    try:
        user_data = auth_service.authenticate_token(credentials.credentials)
        if user_data:
            return AuthenticatedUser.from_dict(user_data)
    except Exception:
        pass

    return None


def require_role(required_role: str):
    """
    Factory for role requirement dependency.

    Args:
        required_role: The role required to access the endpoint.

    Returns:
        Dependency function that validates user has required role.
    """
    async def _require_role(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        if required_role not in user.roles:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required role: {required_role}",
            )
        return user

    return _require_role


def require_any_role(*roles: str):
    """
    Factory for dependency requiring any of the specified roles.

    Args:
        roles: Roles, any of which grants access.

    Returns:
        Dependency function that validates user has at least one role.
    """
    async def _require_any_role(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        if not any(role in user.roles for role in roles):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required one of: {', '.join(roles)}",
            )
        return user

    return _require_any_role


# ============================================================================
# Service Dependencies
# ============================================================================

async def get_security_monitor() -> SecurityMonitor:
    """Get the security monitor instance.

    This dependency is async to avoid FastAPI running it in a threadpool, which
    can deadlock under certain middleware/test transports.
    """
    return security_monitor
