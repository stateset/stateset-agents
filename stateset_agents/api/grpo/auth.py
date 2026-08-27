"""Authentication helpers for the GRPO API service."""

from __future__ import annotations

import hmac
import uuid

from fastapi import HTTPException, Request

from stateset_agents.utils.credentials import credential_fingerprint

from ..logging_config import get_logger
from .config import get_grpo_config
from .metrics import get_grpo_metrics
from .rate_limiter import get_rate_limiter

logger = get_logger(__name__)


class RequestContext:
    """Authenticated request context."""

    def __init__(
        self,
        request_id: str,
        user_id: str,
        roles: list[str],
        api_key: str | None,
        client: str,
    ) -> None:
        self.request_id = request_id
        self.user_id = user_id
        self.roles = roles
        self.api_key = api_key
        self.client = client


def _extract_api_key(request: Request) -> str | None:
    """Extract API key from request headers."""
    auth_header = request.headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    api_key = request.headers.get("x-api-key")
    return api_key.strip() if api_key else None


def _derive_api_user_id(api_key: str) -> str:
    """Derive a stable, non-spoofable user id from API key material."""
    return f"api_key:{credential_fingerprint(api_key)}"


def _derive_anonymous_user_id(client_ip: str) -> str:
    """Derive an anonymous identity from client network identity."""
    return f"anonymous:{credential_fingerprint(client_ip)}"


async def verify_request(request: Request) -> RequestContext:
    """Authenticate request and enforce rate limits."""
    config = get_grpo_config()
    rate_limiter = get_rate_limiter()
    metrics = get_grpo_metrics()

    client_ip = request.client.host if request.client else "unknown"
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    api_key = _extract_api_key(request)
    credential_id: str | None = None

    if config.api_keys:
        matched_key = (
            next(
                (
                    stored_key
                    for stored_key in config.api_keys
                    if api_key is not None and hmac.compare_digest(stored_key, api_key)
                ),
                None,
            )
            if api_key
            else None
        )
        if matched_key is None:
            raise HTTPException(
                status_code=401,
                detail="A valid API key is required for this API",
            )
        configured_roles = config.api_keys[matched_key]
        roles = [role for role in configured_roles if role != "admin"] or ["user"]
        if "admin" in configured_roles:
            logger.debug(
                "Ignoring admin role for API-key authenticated request",
                extra={
                    "client_ip": client_ip,
                    "path": str(request.url.path),
                },
            )
        if request.headers.get("x-user-id"):
            logger.debug(
                "Ignoring untrusted X-User-ID header for API-key authenticated request",
                extra={
                    "client_ip": client_ip,
                    "path": str(request.url.path),
                    "provided_user_id": request.headers.get("x-user-id"),
                },
            )
        user_id = _derive_api_user_id(matched_key)
        credential_id = credential_fingerprint(matched_key)
    elif config.allow_anonymous:
        roles = ["anonymous"]
        if request.headers.get("x-user-id"):
            logger.debug(
                "Ignoring untrusted X-User-ID header for anonymous access",
                extra={
                    "client_ip": client_ip,
                    "path": str(request.url.path),
                    "provided_user_id": request.headers.get("x-user-id"),
                },
            )
        user_id = _derive_anonymous_user_id(client_ip)
    else:
        raise HTTPException(
            status_code=401,
            detail="API key required. Set GRPO_ALLOW_ANONYMOUS=true to enable unauthenticated access.",
        )

    limit_key = api_key or client_ip
    if not rate_limiter.allow(limit_key, config.rate_limit_per_minute):
        metrics.record_rate_limit_hit()
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please retry after a short delay.",
        )

    return RequestContext(
        request_id=request_id,
        user_id=user_id,
        roles=roles,
        api_key=credential_id,
        client=client_ip,
    )
