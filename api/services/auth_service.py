from typing import Dict, Optional
from utils.security import AuthService
from ..config import get_config

# Initialize with config
config = get_config()

# Ensure secret is present (config validation handles this, but safe to default for type checkers)
secret_key = config.security.jwt_secret or "temporary-secret-for-dev"
auth_service = AuthService(secret_key)

def get_auth_service() -> AuthService:
    return auth_service
