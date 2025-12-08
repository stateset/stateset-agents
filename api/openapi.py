"""
OpenAPI Documentation Module

Enhanced OpenAPI schema customization, documentation, and examples
for professional API documentation.
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


# ============================================================================
# API Documentation Metadata
# ============================================================================

API_TITLE = "StateSet Agents API"
API_VERSION = "2.0.0"
API_DESCRIPTION = """
## StateSet Agents API

A comprehensive REST API for training and deploying AI agents using
reinforcement learning techniques.

### Key Features

| Feature | Description |
|---------|-------------|
| **Agent Management** | Create, configure, and manage AI agents |
| **Conversations** | Interactive multi-turn conversations with agents |
| **Training** | Train agents using GRPO and other RL algorithms |
| **Monitoring** | Real-time metrics, logging, and observability |
| **Security** | JWT authentication, rate limiting, prompt injection detection |

### Authentication

The API supports multiple authentication methods:

- **Bearer Token**: `Authorization: Bearer <token>`
- **API Key**: `X-API-Key: <api_key>`

### Rate Limiting

Default rate limits:
- **60 requests/minute** per API key
- **30 requests/minute** for unauthenticated requests

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Unix timestamp when limit resets

### Error Handling

All errors follow a consistent format:

```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human-readable message",
        "details": [...]
    },
    "request_id": "uuid",
    "timestamp": "ISO-8601",
    "path": "/api/endpoint"
}
```

### Versioning

The API uses URL-based versioning. Current version: **v1**

Deprecated endpoints will include a `Deprecation` header with the sunset date.
"""

API_CONTACT = {
    "name": "StateSet API Support",
    "url": "https://github.com/stateset/stateset-agents",
    "email": "api-support@stateset.io",
}

API_LICENSE = {
    "name": "MIT",
    "url": "https://opensource.org/licenses/MIT",
}

API_TAGS = [
    {
        "name": "agents",
        "description": "Agent management operations - create, configure, and manage AI agents.",
        "externalDocs": {
            "description": "Agent documentation",
            "url": "https://docs.stateset.io/agents",
        },
    },
    {
        "name": "conversations",
        "description": "Conversation management - interactive multi-turn dialogues with agents.",
        "externalDocs": {
            "description": "Conversation documentation",
            "url": "https://docs.stateset.io/conversations",
        },
    },
    {
        "name": "training",
        "description": "Training job management - start, monitor, and control agent training.",
        "externalDocs": {
            "description": "Training documentation",
            "url": "https://docs.stateset.io/training",
        },
    },
    {
        "name": "metrics",
        "description": "System metrics and monitoring - health checks, performance metrics.",
    },
    {
        "name": "health",
        "description": "Health check endpoints for monitoring and load balancers.",
    },
]


# ============================================================================
# Response Examples
# ============================================================================

RESPONSE_EXAMPLES = {
    "agent_created": {
        "summary": "Agent Created Successfully",
        "value": {
            "agent_id": "agent-123e4567-e89b",
            "created_at": "2024-01-15T10:30:00Z",
            "config": {
                "model_name": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            "message": "Agent created successfully",
        },
    },
    "conversation_response": {
        "summary": "Conversation Response",
        "value": {
            "conversation_id": "conv-789abc",
            "response": "Hello! How can I assist you today?",
            "tokens_used": 25,
            "processing_time": 0.234,
        },
    },
    "training_started": {
        "summary": "Training Job Started",
        "value": {
            "training_id": "train-456def",
            "status": "running",
            "message": "Training started successfully",
            "estimated_time": 3600,
        },
    },
    "validation_error": {
        "summary": "Validation Error",
        "value": {
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": [
                    {
                        "field": "body.prompts",
                        "message": "At least one prompt is required",
                        "code": "value_error.list.min_items",
                    }
                ],
            },
            "request_id": "req-123",
            "timestamp": "2024-01-15T10:30:00Z",
            "path": "/api/v1/training",
        },
    },
    "not_found": {
        "summary": "Resource Not Found",
        "value": {
            "error": {
                "code": "NOT_FOUND",
                "message": "Agent 'agent-123' not found",
            },
            "request_id": "req-456",
            "timestamp": "2024-01-15T10:30:00Z",
            "path": "/api/v1/agents/agent-123",
        },
    },
    "rate_limited": {
        "summary": "Rate Limit Exceeded",
        "value": {
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests. Please retry after 60 seconds.",
            },
            "request_id": "req-789",
            "timestamp": "2024-01-15T10:30:00Z",
            "path": "/api/v1/conversations",
        },
    },
    "unauthorized": {
        "summary": "Authentication Required",
        "value": {
            "error": {
                "code": "UNAUTHORIZED",
                "message": "Authentication required. Please provide a valid API key.",
            },
            "request_id": "req-abc",
            "timestamp": "2024-01-15T10:30:00Z",
            "path": "/api/v1/agents",
        },
    },
}


# ============================================================================
# Security Schemes
# ============================================================================

SECURITY_SCHEMES = {
    "bearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "JWT Bearer token authentication",
    },
    "apiKeyAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "API key authentication via header",
    },
}


# ============================================================================
# Custom OpenAPI Schema Generator
# ============================================================================

def custom_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """
    Generate custom OpenAPI schema with enhanced documentation.

    This function creates a comprehensive OpenAPI specification with:
    - Detailed API description and metadata
    - Tag descriptions with external documentation links
    - Security scheme definitions
    - Response examples
    - Server definitions for different environments
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
        tags=API_TAGS,
    )

    # Add contact and license info
    openapi_schema["info"]["contact"] = API_CONTACT
    openapi_schema["info"]["license"] = API_LICENSE
    openapi_schema["info"]["x-logo"] = {
        "url": "https://stateset.io/logo.png",
        "altText": "StateSet Logo",
    }

    # Add terms of service
    openapi_schema["info"]["termsOfService"] = "https://stateset.io/terms"

    # Add server definitions
    openapi_schema["servers"] = [
        {
            "url": "https://api.stateset.io",
            "description": "Production server",
        },
        {
            "url": "https://staging-api.stateset.io",
            "description": "Staging server",
        },
        {
            "url": "http://localhost:8000",
            "description": "Local development server",
        },
    ]

    # Add security schemes
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    openapi_schema["components"]["securitySchemes"] = SECURITY_SCHEMES

    # Add global security requirement
    openapi_schema["security"] = [
        {"bearerAuth": []},
        {"apiKeyAuth": []},
    ]

    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "Full API Documentation",
        "url": "https://docs.stateset.io/api",
    }

    # Enhance paths with additional metadata
    _enhance_paths(openapi_schema)

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def _enhance_paths(schema: Dict[str, Any]) -> None:
    """Enhance path definitions with additional metadata."""
    paths = schema.get("paths", {})

    for path, methods in paths.items():
        for method, operation in methods.items():
            if not isinstance(operation, dict):
                continue

            # Add operation ID if missing
            if "operationId" not in operation:
                operation["operationId"] = _generate_operation_id(path, method)

            # Add x-code-samples for common operations
            if "x-code-samples" not in operation:
                operation["x-code-samples"] = _generate_code_samples(path, method)

            # Enhance response descriptions
            _enhance_responses(operation)


def _generate_operation_id(path: str, method: str) -> str:
    """Generate a readable operation ID from path and method."""
    # Convert path to camelCase operation name
    parts = path.strip("/").split("/")
    parts = [p for p in parts if not p.startswith("{")]

    if parts:
        name = "_".join(parts)
        return f"{method}_{name}"

    return f"{method}_root"


def _generate_code_samples(path: str, method: str) -> List[Dict[str, str]]:
    """Generate code samples for different languages."""
    samples = []

    # Python example
    python_sample = f'''import requests

response = requests.{method}(
    "https://api.stateset.io{path}",
    headers={{"Authorization": "Bearer YOUR_TOKEN"}}
)
print(response.json())'''

    samples.append({
        "lang": "Python",
        "source": python_sample,
    })

    # cURL example
    curl_sample = f'''curl -X {method.upper()} \\
  "https://api.stateset.io{path}" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -H "Content-Type: application/json"'''

    samples.append({
        "lang": "cURL",
        "source": curl_sample,
    })

    # JavaScript example
    js_sample = f'''const response = await fetch("https://api.stateset.io{path}", {{
  method: "{method.upper()}",
  headers: {{
    "Authorization": "Bearer YOUR_TOKEN",
    "Content-Type": "application/json"
  }}
}});
const data = await response.json();
console.log(data);'''

    samples.append({
        "lang": "JavaScript",
        "source": js_sample,
    })

    return samples


def _enhance_responses(operation: Dict[str, Any]) -> None:
    """Enhance response definitions with examples and descriptions."""
    responses = operation.get("responses", {})

    # Add common response examples if not present
    common_responses = {
        "401": {
            "description": "Authentication required",
            "content": {
                "application/json": {
                    "example": RESPONSE_EXAMPLES["unauthorized"]["value"],
                }
            },
        },
        "422": {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "example": RESPONSE_EXAMPLES["validation_error"]["value"],
                }
            },
        },
        "429": {
            "description": "Rate limit exceeded",
            "content": {
                "application/json": {
                    "example": RESPONSE_EXAMPLES["rate_limited"]["value"],
                }
            },
        },
    }

    for code, response in common_responses.items():
        if code not in responses:
            responses[code] = response


def setup_openapi(app: FastAPI) -> None:
    """Set up custom OpenAPI schema for the application."""
    app.openapi = lambda: custom_openapi_schema(app)


# ============================================================================
# API Documentation Endpoints
# ============================================================================

def add_documentation_routes(app: FastAPI) -> None:
    """Add additional documentation routes."""

    @app.get(
        "/api/docs/changelog",
        tags=["documentation"],
        summary="API Changelog",
        description="Get the API changelog with version history.",
    )
    async def get_changelog():
        """Get API changelog."""
        return {
            "versions": [
                {
                    "version": "2.0.0",
                    "date": "2024-01-15",
                    "changes": [
                        "Added distributed caching support",
                        "Enhanced security with prompt injection detection",
                        "Added circuit breaker patterns",
                        "Improved rate limiting with sliding window",
                    ],
                },
                {
                    "version": "1.5.0",
                    "date": "2024-01-01",
                    "changes": [
                        "Added multi-turn conversation support",
                        "Improved training job management",
                        "Added comprehensive metrics",
                    ],
                },
                {
                    "version": "1.0.0",
                    "date": "2023-12-01",
                    "changes": [
                        "Initial release",
                        "Agent management endpoints",
                        "Basic training support",
                    ],
                },
            ],
        }

    @app.get(
        "/api/docs/errors",
        tags=["documentation"],
        summary="Error Codes Reference",
        description="Get a reference of all possible error codes.",
    )
    async def get_error_codes():
        """Get error codes reference."""
        return {
            "error_codes": {
                "BAD_REQUEST": {
                    "status": 400,
                    "description": "The request was malformed or invalid.",
                },
                "UNAUTHORIZED": {
                    "status": 401,
                    "description": "Authentication is required.",
                },
                "FORBIDDEN": {
                    "status": 403,
                    "description": "You don't have permission to access this resource.",
                },
                "NOT_FOUND": {
                    "status": 404,
                    "description": "The requested resource was not found.",
                },
                "VALIDATION_ERROR": {
                    "status": 422,
                    "description": "The request body failed validation.",
                },
                "RATE_LIMIT_EXCEEDED": {
                    "status": 429,
                    "description": "Too many requests. Please retry later.",
                },
                "INTERNAL_ERROR": {
                    "status": 500,
                    "description": "An unexpected error occurred.",
                },
            },
        }

    @app.get(
        "/api/docs/rate-limits",
        tags=["documentation"],
        summary="Rate Limits",
        description="Get rate limit information.",
    )
    async def get_rate_limits():
        """Get rate limit information."""
        return {
            "rate_limits": {
                "default": {
                    "requests_per_minute": 60,
                    "description": "Standard rate limit for authenticated requests",
                },
                "unauthenticated": {
                    "requests_per_minute": 30,
                    "description": "Rate limit for unauthenticated requests",
                },
                "training": {
                    "concurrent_jobs": 5,
                    "description": "Maximum concurrent training jobs per user",
                },
            },
            "headers": {
                "X-RateLimit-Limit": "Maximum requests allowed in window",
                "X-RateLimit-Remaining": "Requests remaining in current window",
                "X-RateLimit-Reset": "Unix timestamp when window resets",
                "Retry-After": "Seconds to wait before retrying (on 429)",
            },
        }
