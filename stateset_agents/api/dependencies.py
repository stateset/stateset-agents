"""
API Dependencies Module

Dependency injection for authentication, authorization, and common services.
"""

import hashlib
import time

from fastapi import Depends, HTTPException, Request

from stateset_agents.utils.security import SecurityMonitor

from .auth import AuthenticatedUser, authenticate_request
from .errors import UnauthorizedError
from .security import get_api_security_monitor


# ============================================================================
# Security Setup
# ============================================================================

security_monitor = SecurityMonitor()

AUTH_EXCEPTIONS = (RuntimeError, TypeError, ValueError)


def _get_client_ip(request: Request) -> str:
    """Return client IP for security tracking."""
    client = request.client
    if client is None:
        return "unknown"
    return str(client.host)


def _get_lockout_key(request: Request) -> str:
    """Return stable lockout key for the presented credentials."""
    client_ip = _get_client_ip(request)

    # Prefer API key / bearer token for credential-scoped lockout.
    api_key = request.headers.get("X-API-Key")
    if api_key:
        raw_key = api_key.strip()
    else:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            raw_key = auth_header.split(" ", 1)[1].strip()
        else:
            raw_key = ""

    if raw_key:
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
        return f"cred:{client_ip}:{digest}"

    # Fall back to IP for unauthenticated requests.
    return f"ip:{client_ip}"


def _clear_client_lockouts(api_security, request: Request) -> None:
    """Clear tracked failures for the authenticated client after a success."""
    client_ip = _get_client_ip(request)
    key_prefixes = (f"cred:{client_ip}:", f"ip:{client_ip}")

    for key in list(api_security.auth_tracker.failures):
        if key.startswith(key_prefixes):
            api_security.auth_tracker.clear_failures(key)

    for key in list(api_security.auth_tracker.lockouts):
        if key.startswith(key_prefixes):
            api_security.auth_tracker.clear_failures(key)


def _lockout_retry_after(api_security, lockout_key: str) -> int:
    """Get remaining lockout seconds for a key, if locked out."""
    lockout_until = api_security.auth_tracker.lockouts.get(lockout_key)
    if not lockout_until:
        return 0

    remaining_seconds = int(lockout_until - time.time())
    if remaining_seconds <= 0:
        # Clean up expired lockout state.
        api_security.auth_tracker.lockouts.pop(lockout_key, None)
        api_security.auth_tracker.failures.pop(lockout_key, None)
        return 0
    return max(1, remaining_seconds)


# ============================================================================
# Authentication Dependencies
# ============================================================================


async def get_current_user(request: Request) -> AuthenticatedUser:
    """
    FastAPI dependency to get the current authenticated user.

    Returns:
        AuthenticatedUser with user information.

    Raises:
        HTTPException 401: If authentication fails.
    """
    from .config import get_config

    api_security = get_api_security_monitor()
    lockout_key = _get_lockout_key(request)
    config = get_config()

    if not config.security.require_auth:
        return authenticate_request(request)

    # Enforce configured auth lockout before validation.
    retry_after = _lockout_retry_after(api_security, lockout_key)
    if retry_after:
        raise HTTPException(
            status_code=401,
            detail="Authentication temporarily locked due to repeated failures.",
            headers={
                "WWW-Authenticate": "Bearer",
                "Retry-After": str(retry_after),
            },
        )

    try:
        user = authenticate_request(request)

        # Clear any previous failures on successful auth.
        _clear_client_lockouts(api_security, request)
        return user

    except UnauthorizedError as exc:
        locked_out, _ = api_security.log_auth_failure(
            client_ip=lockout_key,
            path=str(request.url.path),
            reason=exc.message,
        )
        headers = {"WWW-Authenticate": "Bearer"}
        if locked_out:
            retry_after = _lockout_retry_after(api_security, lockout_key)
            if retry_after:
                headers["Retry-After"] = str(retry_after)
            detail = (
                "Authentication temporarily locked due to repeated failures. "
                "Please retry later."
                if retry_after
                else exc.message
            )
        else:
            detail = exc.message

        raise HTTPException(status_code=401, detail=detail, headers=headers) from exc

    except AUTH_EXCEPTIONS as e:
        locked_out, _ = api_security.log_auth_failure(
            client_ip=lockout_key,
            path=str(request.url.path),
            reason=str(e),
        )
        headers = {"WWW-Authenticate": "Bearer"}
        if locked_out:
            retry_after = _lockout_retry_after(api_security, lockout_key)
            if retry_after:
                headers["Retry-After"] = str(retry_after)
            detail = (
                "Authentication temporarily locked due to repeated failures. "
                "Please retry later."
                if retry_after
                else "Authentication failed"
            )
        else:
            detail = "Authentication failed"

        security_monitor.log_security_event(
            "authentication_failure",
            {"error": str(e), "path": str(request.url.path)},
        )
        raise HTTPException(
            status_code=401,
            detail=detail,
            headers=headers,
        ) from e


async def get_optional_user(request: Request) -> AuthenticatedUser | None:
    """
    Get current user if authenticated, None otherwise.

    Useful for endpoints that work both authenticated and anonymously.
    """
    from .config import get_config

    config = get_config()
    if not config.security.require_auth:
        return None

    try:
        return authenticate_request(request)
    except AUTH_EXCEPTIONS:
        pass
    except UnauthorizedError:
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

    async def _require_role(
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> AuthenticatedUser:
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

    async def _require_any_role(
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> AuthenticatedUser:
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


def get_inference_service(request: Request):
    """Get the InferenceService from app state.

    The service is created during lifespan in ``main.py`` and stored on
    ``app.state.inference_service``.  If it hasn't been created yet (e.g.
    during isolated tests), a fresh instance is created from environment
    variables.
    """
    from .services.inference_service import InferenceConfig, InferenceService

    service = getattr(request.app.state, "inference_service", None)
    if service is None:
        service = InferenceService(InferenceConfig.from_env())
        request.app.state.inference_service = service
    return service


def get_agent_service(request: Request):
    """Get the AgentService from app state.

    Created during lifespan in ``main.py`` and stored on
    ``app.state.agent_service``.
    """
    from .services.agent_service import AgentService

    service = getattr(request.app.state, "agent_service", None)
    if service is None:
        service = AgentService(security_monitor)
        request.app.state.agent_service = service
    return service


def get_training_service(request: Request):
    """Get the TrainingService from app state.

    Created during lifespan in ``main.py`` and stored on
    ``app.state.training_service``.
    """
    from .services.training_service import TrainingService

    service = getattr(request.app.state, "training_service", None)
    if service is None:
        service = TrainingService()
        request.app.state.training_service = service
    return service


async def require_auth_if_enabled(request: Request) -> AuthenticatedUser | None:
    """Raise 401 when authentication is required but no user is present.

    Use as a dependency in endpoints that are optionally authenticated: when
    the server's ``security.require_auth`` flag is set, unauthenticated
    requests are rejected.  Otherwise the (possibly ``None``) user is passed
    through.
    """
    from .config import get_config

    config = get_config()
    if config.security.require_auth:
        return await get_current_user(request)

    return await get_optional_user(request)
