from typing import Dict
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .services.auth_service import get_auth_service, AuthService
from utils.security import SecurityMonitor

security = HTTPBearer()
security_monitor = SecurityMonitor()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> Dict:
    """Get current authenticated user."""
    token = credentials.credentials

    try:
        user_data = auth_service.authenticate_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return user_data
    except Exception as e:
        security_monitor.log_security_event(
            "authentication_failure",
            {"error": str(e), "token_prefix": token[:10] if token else "None"},
        )
        raise HTTPException(status_code=401, detail="Authentication failed")

def require_role(required_role: str):
    """Factory for role requirement dependency."""
    async def _require_role(user_data: Dict = Depends(get_current_user)):
        user_roles = user_data.get("roles", [])
        if required_role not in user_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required role: {required_role}",
            )
        return user_data
    return _require_role

def get_security_monitor() -> SecurityMonitor:
    return security_monitor
