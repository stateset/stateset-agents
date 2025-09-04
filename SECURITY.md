# Security Policy

## Supported Versions

We take security seriously and actively maintain security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :x:                |
| < 0.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in StateSet Agents, please help us by reporting it responsibly.

### How to Report

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing:
- **security@stateset.ai**

Include the following information in your report:
- A clear description of the vulnerability
- Steps to reproduce the issue
- Potential impact and severity
- Any suggested fixes or mitigations

### Response Timeline

- **Initial Response**: Within 24 hours
- **Vulnerability Assessment**: Within 72 hours  
- **Fix Development**: Within 1-2 weeks for critical issues
- **Public Disclosure**: After fix is deployed and tested

## Security Best Practices

### For Contributors

1. **Input Validation**: Always validate and sanitize user inputs
2. **Secure Dependencies**: Keep dependencies updated and scan for vulnerabilities
3. **Secure Coding**: Follow secure coding practices (see guidelines below)
4. **Secrets Management**: Never commit secrets or sensitive data
5. **Access Control**: Implement proper authentication and authorization

### For Users

1. **Regular Updates**: Keep the framework and dependencies updated
2. **Secure Configuration**: Use strong passwords and secure API keys
3. **Network Security**: Deploy behind firewalls and use HTTPS
4. **Monitoring**: Enable logging and monitoring for security events
5. **Backup Security**: Encrypt backups and secure backup storage

## Security Features

### Built-in Security Features

- **Input Sanitization**: Automatic input validation and sanitization
- **Rate Limiting**: Built-in rate limiting for API endpoints
- **Authentication**: JWT-based authentication system
- **Authorization**: Role-based access control (RBAC)
- **Audit Logging**: Comprehensive security event logging
- **Encryption**: Data encryption at rest and in transit

### Security Scanning

We use multiple security scanning tools:

- **Bandit**: Python security linter
- **Safety**: Dependency vulnerability scanner  
- **Trivy**: Container vulnerability scanning
- **Snyk**: Code and dependency analysis

## Secure Coding Guidelines

### General Principles

1. **Defense in Depth**: Multiple layers of security controls
2. **Fail-Safe Defaults**: Secure defaults with explicit opt-in for risky features
3. **Least Privilege**: Minimum required permissions
4. **Input Validation**: Validate all inputs at every layer
5. **Output Encoding**: Encode all outputs to prevent injection attacks

### Python-Specific Guidelines

#### Input Validation
```python
from typing import Union
import re

def validate_email(email: str) -> bool:
    """Validate email format and prevent injection."""
    if not isinstance(email, str) or len(email) > 254:
        return False
    
    # Use regex that prevents ReDoS attacks
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return bool(email_pattern.match(email))

def sanitize_input(input_str: str, max_length: int = 1000) -> str:
    """Sanitize user input."""
    if not isinstance(input_str, str):
        raise ValueError("Input must be a string")
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>]', '', input_str)
    return sanitized[:max_length]
```

#### Secure File Operations
```python
import os
from pathlib import Path

def secure_file_upload(file_path: str, allowed_extensions: set) -> bool:
    """Securely handle file uploads."""
    path = Path(file_path)
    
    # Validate file extension
    if path.suffix.lower() not in allowed_extensions:
        return False
    
    # Prevent directory traversal
    resolved_path = path.resolve()
    upload_dir = Path("/safe/upload/dir").resolve()
    
    if not str(resolved_path).startswith(str(upload_dir)):
        return False
    
    # Additional security checks
    if path.exists() and path.stat().st_size > 10 * 1024 * 1024:  # 10MB limit
        return False
    
    return True
```

#### Authentication & Authorization
```python
from datetime import datetime, timedelta
import jwt
import bcrypt

class AuthService:
    """Secure authentication service."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.bcrypt_rounds = 12  # High work factor for security
    
    def hash_password(self, password: str) -> str:
        """Securely hash passwords."""
        salt = bcrypt.gensalt(rounds=self.bcrypt_rounds)
        return bcrypt.hashpw(password.encode(), salt).decode()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode(), hashed.encode())
    
    def generate_token(self, user_id: str, roles: list) -> str:
        """Generate secure JWT token."""
        payload = {
            'user_id': user_id,
            'roles': roles,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
```

### API Security
```python
from fastapi import HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

@limiter.limit("100/minute")
def rate_limited_endpoint(request: Request):
    """Rate-limited API endpoint."""
    pass

def validate_api_key(api_key: str) -> bool:
    """Validate API key format and existence."""
    if not api_key or len(api_key) < 32:
        return False
    
    # Check against secure key store (not hardcoded!)
    # This is just an example - use proper key management
    return api_key in get_valid_api_keys()
```

## Security Testing

### Unit Tests for Security
```python
import pytest
from unittest.mock import patch

def test_input_validation():
    """Test input validation functions."""
    # Test valid inputs
    assert validate_email("user@example.com")
    
    # Test invalid inputs
    assert not validate_email("invalid-email")
    assert not validate_email("a" * 300 + "@example.com")  # Too long
    
    # Test sanitization
    assert sanitize_input("<script>alert('xss')</script>") == "scriptalert('xss')/script"

def test_authentication_security():
    """Test authentication security."""
    auth = AuthService("test_secret_key")
    
    # Test password hashing
    hashed = auth.hash_password("test_password")
    assert auth.verify_password("test_password", hashed)
    assert not auth.verify_password("wrong_password", hashed)
    
    # Test token generation and verification
    token = auth.generate_token("user123", ["user"])
    payload = auth.verify_token(token)
    assert payload["user_id"] == "user123"
    assert "user" in payload["roles"]
```

## Incident Response

### Security Incident Response Plan

1. **Detection**: Monitor for security events and anomalies
2. **Assessment**: Evaluate impact and severity of incidents
3. **Containment**: Isolate affected systems and stop attacks
4. **Recovery**: Restore systems and data from backups
5. **Lessons Learned**: Document findings and improve processes

### Contact Information

- **Security Team**: security@stateset.ai
- **Emergency**: +1-555-0123 (24/7 emergency line)
- **PGP Key**: Available at https://stateset.ai/security/pgp.txt

## Security Updates

We recommend subscribing to our security mailing list for important security updates:

- **Mailing List**: security-announce@stateset.ai
- **RSS Feed**: https://stateset.ai/security/rss.xml
- **Advisories**: https://stateset.ai/security/advisories/

## Acknowledgments

We appreciate the security research community for their contributions to keeping open source software secure. Security researchers who responsibly disclose vulnerabilities will be acknowledged in our security advisories.
