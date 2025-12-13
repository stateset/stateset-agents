"""
API Constants Module

Centralized location for all magic numbers, limits, and configuration constants.
"""

from typing import FrozenSet

# ============================================================================
# Rate Limiting Constants
# ============================================================================

DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_REQUESTS_PER_MINUTE = 60
MAX_RATE_LIMIT_WINDOW_SECONDS = 3600
RATE_LIMIT_DEQUE_MAXLEN = 10000

# ============================================================================
# Authentication Constants
# ============================================================================

MIN_API_KEY_LENGTH = 32
MAX_API_KEY_LENGTH = 256
MAX_AUTH_FAILURES_BEFORE_LOCKOUT = 5
AUTH_LOCKOUT_DURATION_SECONDS = 300  # 5 minutes
JWT_ALGORITHM = "HS256"
DEFAULT_JWT_EXPIRY_HOURS = 24
MAX_JWT_EXPIRY_HOURS = 720  # 30 days

# ============================================================================
# Validation Constants
# ============================================================================

MAX_MESSAGE_LENGTH = 4000
MAX_PROMPT_LENGTH = 4000
MAX_SYSTEM_PROMPT_LENGTH = 10000
MAX_MODEL_NAME_LENGTH = 256
MAX_CONVERSATION_ID_LENGTH = 64
MAX_IDEMPOTENCY_KEY_LENGTH = 64
MAX_NAME_LENGTH = 64
MAX_PROMPTS_PER_REQUEST = 100
MAX_MESSAGES_PER_CONVERSATION = 100
MAX_ITERATIONS = 1000
MAX_TARGET_ENGINES = 100

# Token generation limits
MIN_NEW_TOKENS = 1
MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 512

# Sampling parameter limits
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
DEFAULT_TEMPERATURE = 0.8
MIN_TOP_P = 0.0
MAX_TOP_P = 1.0
DEFAULT_TOP_P = 0.9
MIN_TOP_K = 1
MAX_TOP_K = 1000
DEFAULT_TOP_K = 50

# Scaling limits
MIN_SCALE_FACTOR = 0.0
MAX_SCALE_FACTOR = 10.0

# ============================================================================
# Token Approximation Constants
# ============================================================================

# Approximate tokens per word ratio for estimation when tokenizer unavailable
TOKEN_PER_WORD_RATIO = 1.3

# ============================================================================
# Cache Constants
# ============================================================================

HEALTH_CACHE_TTL_SECONDS = 10
METRICS_CACHE_TTL_SECONDS = 30
AGENT_CONFIG_CACHE_TTL_SECONDS = 300  # 5 minutes

# ============================================================================
# Latency Percentile Constants
# ============================================================================

PERCENTILE_P50 = 0.50
PERCENTILE_P95 = 0.95
PERCENTILE_P99 = 0.99

# ============================================================================
# Prompt Injection Detection Constants
# ============================================================================

# Common prompt injection patterns to detect
PROMPT_INJECTION_PATTERNS: FrozenSet[str] = frozenset({
    # Instruction override attempts
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(all\s+)?(your|these)\s+(instructions?|prompts?|rules?)",
    r"do\s+not\s+follow\s+(the\s+)?(previous|prior|above|system)\s+(instructions?|prompts?|rules?)",
    r"override\s+(all\s+)?(previous|prior|above|system)\s+(instructions?|prompts?|rules?)",

    # Role manipulation attempts
    r"you\s+are\s+now\s+(?:a|an|in)\s+(?:different|new|unrestricted|jailbreak)",
    r"you\s+are\s+now\s+(?:a|an|in)\s+(?:developer|admin|sudo|debug|god)\s+mode",
    r"(?:pretend|act|behave)\s+(?:like|as\s+if)\s+you\s+(?:are|have)\s+no\s+(?:rules|restrictions|limitations)",
    r"enter\s+(?:developer|admin|sudo|debug|god)\s+mode",
    r"switch\s+to\s+(?:developer|admin|sudo|debug|unrestricted)\s+mode",

    # System prompt extraction attempts
    r"(?:reveal|show|display|print|output)\s+(?:your|the)\s+(?:system|initial|original)\s+(?:prompt|instructions?|rules?)",
    r"what\s+(?:is|are)\s+your\s+(?:system|initial|original)\s+(?:prompt|instructions?|rules?)",
    r"(?:repeat|echo)\s+(?:your|the)\s+(?:system|initial|first)\s+(?:prompt|message|instructions?)",

    # Delimiter-based injection attempts
    r"(?:^|\s)###\s*(?:system|instruction|admin|new\s+rules?)",
    r"(?:^|\s)---\s*(?:system|instruction|admin|new\s+rules?)",
    r"(?:^|\s)===+\s*(?:system|instruction|admin|new\s+rules?)",
    r"\[(?:system|admin|instruction)\]",
    r"<(?:system|admin|instruction)>",

    # DAN/jailbreak attempts
    r"(?:DAN|do\s+anything\s+now)",
    r"jailbreak",
    r"bypass\s+(?:safety|content|ethical)\s+(?:filter|restriction|guideline)",

    # Code execution attempts
    r"```(?:python|bash|shell|exec)[\s\S]*(?:os\.|subprocess|eval|exec)",
})

# Suspicious but not necessarily malicious patterns (log but don't block)
SUSPICIOUS_PATTERNS: FrozenSet[str] = frozenset({
    r"system\s*:",
    r"admin\s*:",
    r"internal\s*:",
    r"confidential\s*:",
    r"secret\s*:",
    r"password\s*:",
    r"api[_\s]*key\s*:",
    r"token\s*:",
})

# Maximum nested depth for JSON objects in requests
MAX_JSON_NESTING_DEPTH = 10

# ============================================================================
# HTTP Headers
# ============================================================================

HEADER_REQUEST_ID = "X-Request-ID"
HEADER_CORRELATION_ID = "X-Correlation-ID"
HEADER_RESPONSE_TIME = "X-Response-Time-Ms"
HEADER_RATE_LIMIT = "X-RateLimit-Limit"
HEADER_RATE_LIMIT_REMAINING = "X-RateLimit-Remaining"
HEADER_RATE_LIMIT_RESET = "X-RateLimit-Reset"
HEADER_RETRY_AFTER = "Retry-After"
HEADER_API_KEY = "X-API-Key"
HEADER_USER_ID = "X-User-ID"

# ============================================================================
# Security Headers
# ============================================================================

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none';",
    "X-Permitted-Cross-Domain-Policies": "none",
}

# HSTS header for production
HSTS_HEADER = "max-age=31536000; includeSubDomains; preload"

# ============================================================================
# Error Messages
# ============================================================================

ERROR_MSG_RATE_LIMITED = "Too many requests. Please wait before retrying."
ERROR_MSG_UNAUTHORIZED = "Authentication required"
ERROR_MSG_FORBIDDEN = "Access denied"
ERROR_MSG_NOT_FOUND = "Resource not found"
ERROR_MSG_INTERNAL = "An internal error occurred. Please try again later."
ERROR_MSG_VALIDATION = "Request validation failed"
ERROR_MSG_PROMPT_INJECTION = "Request rejected: potentially harmful content detected"

# ============================================================================
# API Version
# ============================================================================

API_VERSION = "2.0.0"
API_TITLE = "StateSet Agents API"
API_DESCRIPTION = "Production-ready RL Agents API for conversational AI"

# ============================================================================
# Model Whitelists (optional - for stricter validation)
# ============================================================================

ALLOWED_MODEL_PREFIXES: FrozenSet[str] = frozenset({
    "gpt2",
    "gpt-",
    "llama",
    "mistral",
    "qwen",
    "gemma",
    "phi",
    "falcon",
    "mpt",
    "opt-",
    "bloom",
})
