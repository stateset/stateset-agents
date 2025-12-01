"""
API Authentication Module

Secure authentication and authorization for API services.
"""

import hashlib
import hmac
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .config import get_config
from .errors import UnauthorizedError, ForbiddenError

logger = logging.getLogger(__name__)

# Security scheme for OpenAPI docs
security_scheme = HTTPBearer(auto_error=False)


# ============================================================================
# Models
# ============================================================================

class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str = Field(..., description="Subject (user ID)")
    roles: List[str] = Field(default_factory=list, description="User roles")
    exp: int = Field(..., description="Expiration timestamp")
    iat: int = Field(..., description="Issued at timestamp")
    jti: str = Field(..., description="JWT ID for revocation")


class AuthenticatedUser(BaseModel):
    """Authenticated user context."""
    user_id: str = Field(..., description="User identifier")
    roles: List[str] = Field(default_factory=list, description="User roles")
    api_key: Optional[str] = Field(None, description="API key used (masked)")
    auth_method: str = Field(..., description="Authentication method used")


# ============================================================================
# JWT Implementation (without external dependencies)
# ============================================================================

class JWTHandler:
    """Simple JWT handler using HMAC-SHA256."""

    def __init__(self, secret: str, algorithm: str = "HS256"):
        self.secret = secret.encode()
        self.algorithm = algorithm

    def _base64url_encode(self, data: bytes) -> str:
        """Base64 URL-safe encoding without padding."""
        import base64
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    def _base64url_decode(self, data: str) -> bytes:
        """Base64 URL-safe decoding with padding restoration."""
        import base64
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    def encode(self, payload: Dict[str, Any]) -> str:
        """Encode a JWT token."""
        import json

        header = {"alg": self.algorithm, "typ": "JWT"}

        # Encode header and payload
        header_b64 = self._base64url_encode(json.dumps(header).encode())
        payload_b64 = self._base64url_encode(json.dumps(payload).encode())

        # Create signature
        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(self.secret, message.encode(), hashlib.sha256).digest()
        signature_b64 = self._base64url_encode(signature)

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def decode(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode and verify a JWT token."""
        import json

        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature_b64 = parts

            # Verify signature
            message = f"{header_b64}.{payload_b64}"
            expected_signature = hmac.new(self.secret, message.encode(), hashlib.sha256).digest()
            actual_signature = self._base64url_decode(signature_b64)

            if not hmac.compare_digest(expected_signature, actual_signature):
                logger.warning("JWT signature verification failed")
                return None

            # Decode payload
            payload_json = self._base64url_decode(payload_b64).decode()
            payload = json.loads(payload_json)

            # Check expiration
            if "exp" in payload and payload["exp"] < time.time():
                logger.warning("JWT token expired")
                return None

            return payload

        except Exception as e:
            logger.warning(f"JWT decode error: {e}")
            return None


# Global JWT handler (initialized lazily)
_jwt_handler: Optional[JWTHandler] = None


def get_jwt_handler() -> JWTHandler:
    """Get the global JWT handler."""
    global _jwt_handler
    if _jwt_handler is None:
        config = get_config()
        if not config.security.jwt_secret:
            raise RuntimeError("JWT secret not configured")
        _jwt_handler = JWTHandler(config.security.jwt_secret)
    return _jwt_handler


# ============================================================================
# Token Generation
# ============================================================================

def generate_token(
    user_id: str,
    roles: List[str],
    expiry_hours: Optional[int] = None,
) -> str:
    """Generate a JWT token for a user."""
    config = get_config()
    jwt = get_jwt_handler()

    now = int(time.time())
    expiry = expiry_hours or config.security.jwt_expiry_hours

    payload = {
        "sub": user_id,
        "roles": roles,
        "iat": now,
        "exp": now + (expiry * 3600),
        "jti": secrets.token_urlsafe(16),
    }

    return jwt.encode(payload)


def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)


# ============================================================================
# Authentication Functions
# ============================================================================

def _extract_bearer_token(request: Request) -> Optional[str]:
    """Extract bearer token from Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return None


def _extract_api_key(request: Request) -> Optional[str]:
    """Extract API key from headers."""
    # Check X-API-Key header first
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key.strip()

    # Fall back to Authorization header
    return _extract_bearer_token(request)


def _mask_api_key(api_key: str) -> str:
    """Mask API key for logging (show first 8 and last 4 chars)."""
    if len(api_key) <= 12:
        return "***"
    return f"{api_key[:8]}...{api_key[-4:]}"


def authenticate_request(request: Request) -> AuthenticatedUser:
    """
    Authenticate a request using API key or JWT token.

    Returns:
        AuthenticatedUser with user information

    Raises:
        UnauthorizedError if authentication fails
    """
    config = get_config()

    # Skip auth if not required
    if not config.security.require_auth:
        return AuthenticatedUser(
            user_id="anonymous",
            roles=["anonymous"],
            auth_method="none",
        )

    # Try API key authentication first
    api_key = _extract_api_key(request)
    if api_key:
        if api_key in config.security.api_keys:
            roles = config.security.api_keys[api_key]
            user_id = request.headers.get("X-User-ID", "api-key-user")

            logger.debug(
                "Authenticated via API key",
                extra={
                    "user_id": user_id,
                    "api_key": _mask_api_key(api_key),
                    "roles": roles,
                }
            )

            return AuthenticatedUser(
                user_id=user_id,
                roles=roles,
                api_key=_mask_api_key(api_key),
                auth_method="api_key",
            )

        # Try JWT token
        jwt = get_jwt_handler()
        payload = jwt.decode(api_key)
        if payload:
            return AuthenticatedUser(
                user_id=payload.get("sub", "unknown"),
                roles=payload.get("roles", []),
                auth_method="jwt",
            )

        # Invalid credentials
        logger.warning(
            "Authentication failed: invalid API key or token",
            extra={
                "client_ip": request.client.host if request.client else "unknown",
                "path": request.url.path,
            }
        )
        raise UnauthorizedError("Invalid API key or token")

    # No credentials provided
    raise UnauthorizedError("Authentication required. Provide an API key or bearer token.")


def require_roles(*required_roles: str):
    """
    Dependency that requires the user to have specific roles.

    Usage:
        @app.get("/admin")
        async def admin_endpoint(user: AuthenticatedUser = Depends(require_roles("admin"))):
            ...
    """
    async def role_checker(request: Request) -> AuthenticatedUser:
        user = authenticate_request(request)

        # Check if user has any of the required roles
        if not any(role in user.roles for role in required_roles):
            logger.warning(
                "Authorization failed: insufficient roles",
                extra={
                    "user_id": user.user_id,
                    "user_roles": user.roles,
                    "required_roles": required_roles,
                    "path": request.url.path,
                }
            )
            raise ForbiddenError(
                f"Insufficient permissions. Required role: {' or '.join(required_roles)}"
            )

        return user

    return role_checker


# ============================================================================
# FastAPI Dependencies
# ============================================================================

async def get_current_user(request: Request) -> AuthenticatedUser:
    """
    FastAPI dependency to get the current authenticated user.

    Usage:
        @app.get("/profile")
        async def get_profile(user: AuthenticatedUser = Depends(get_current_user)):
            return {"user_id": user.user_id}
    """
    return authenticate_request(request)


async def get_optional_user(request: Request) -> Optional[AuthenticatedUser]:
    """
    FastAPI dependency for optional authentication.

    Returns None if not authenticated instead of raising an error.

    Usage:
        @app.get("/public")
        async def public_endpoint(user: Optional[AuthenticatedUser] = Depends(get_optional_user)):
            if user:
                return {"message": f"Hello, {user.user_id}"}
            return {"message": "Hello, anonymous"}
    """
    try:
        return authenticate_request(request)
    except UnauthorizedError:
        return None


# Commonly used role dependencies
require_admin = require_roles("admin")
require_trainer = require_roles("admin", "trainer")
require_user = require_roles("admin", "trainer", "user")


# ============================================================================
# Request Context
# ============================================================================

class RequestContext(BaseModel):
    """Full request context including auth and tracing."""
    request_id: str
    user: AuthenticatedUser
    client_ip: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


async def get_request_context(request: Request) -> RequestContext:
    """
    Get full request context including authentication.

    Usage:
        @app.post("/api/train")
        async def train(ctx: RequestContext = Depends(get_request_context)):
            logger.info(f"Training started by {ctx.user.user_id}")
    """
    user = authenticate_request(request)
    request_id = getattr(request.state, "request_id", "unknown")
    client_ip = request.client.host if request.client else "unknown"

    return RequestContext(
        request_id=request_id,
        user=user,
        client_ip=client_ip,
    )
