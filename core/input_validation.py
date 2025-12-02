"""
Input Validation and Security for StateSet Agents

This module provides comprehensive input validation and security measures
to protect against prompt injection, excessive resource consumption, and
other attack vectors targeting LLM-based systems.

Features:
- Prompt injection detection and prevention
- Input length and token limit enforcement
- Content filtering for harmful patterns
- Rate limiting at the agent level
- Sanitization of special characters and control sequences

Example:
    >>> from core.input_validation import SecureInputValidator, SecurityConfig
    >>>
    >>> validator = SecureInputValidator(SecurityConfig(
    ...     max_input_length=10000,
    ...     detect_injection=True,
    ...     block_system_override=True
    ... ))
    >>>
    >>> # Validate user input before sending to agent
    >>> result = validator.validate("Hello, how are you?")
    >>> if result.is_valid:
    ...     response = await agent.generate_response(result.sanitized_input)
"""

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple

logger = logging.getLogger(__name__)


class SecurityRisk(str, Enum):
    """Categories of security risks detected in input."""

    PROMPT_INJECTION = "prompt_injection"
    SYSTEM_OVERRIDE = "system_override"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    EXCESSIVE_LENGTH = "excessive_length"
    EXCESSIVE_TOKENS = "excessive_tokens"
    HARMFUL_CONTENT = "harmful_content"
    CONTROL_CHARACTERS = "control_characters"
    ENCODED_CONTENT = "encoded_content"
    REPETITION_ATTACK = "repetition_attack"


class RiskSeverity(str, Enum):
    """Severity levels for security risks."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityThreat:
    """Represents a detected security threat.

    Attributes:
        risk_type: Category of the threat
        severity: How serious the threat is
        description: Human-readable description
        matched_pattern: The pattern or content that triggered detection
        position: Character position where threat was found
        recommended_action: Suggested response to the threat
    """

    risk_type: SecurityRisk
    severity: RiskSeverity
    description: str
    matched_pattern: Optional[str] = None
    position: Optional[int] = None
    recommended_action: str = "block"


@dataclass
class ValidationResult:
    """Result of input validation.

    Attributes:
        is_valid: Whether the input passed all checks
        sanitized_input: Cleaned version of the input (if valid)
        original_input: The original input before sanitization
        threats: List of detected security threats
        warnings: Non-blocking issues to be aware of
        processing_time_ms: Time taken to validate
    """

    is_valid: bool
    sanitized_input: Optional[str] = None
    original_input: str = ""
    threats: List[SecurityThreat] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def has_critical_threats(self) -> bool:
        """Check if any critical threats were detected."""
        return any(t.severity == RiskSeverity.CRITICAL for t in self.threats)

    def get_threat_summary(self) -> str:
        """Get a summary of all threats."""
        if not self.threats:
            return "No threats detected"
        return "; ".join(f"{t.risk_type.value}: {t.description}" for t in self.threats)


@dataclass
class SecurityConfig:
    """Configuration for input security validation.

    Attributes:
        max_input_length: Maximum character length (default: 32000)
        max_tokens_estimate: Estimated max tokens (chars / 4)
        detect_injection: Enable prompt injection detection
        block_system_override: Block attempts to override system prompts
        block_jailbreaks: Block known jailbreak patterns
        sanitize_control_chars: Remove control characters
        detect_encoding_attacks: Detect base64/unicode abuse
        detect_repetition: Detect repetition-based attacks
        custom_blocked_patterns: Additional regex patterns to block
        custom_allowed_patterns: Patterns to always allow (whitelist)
        rate_limit_requests: Max requests per window
        rate_limit_window_seconds: Rate limit time window
    """

    max_input_length: int = 32000
    max_tokens_estimate: int = 8000
    detect_injection: bool = True
    block_system_override: bool = True
    block_jailbreaks: bool = True
    sanitize_control_chars: bool = True
    detect_encoding_attacks: bool = True
    detect_repetition: bool = True
    custom_blocked_patterns: List[str] = field(default_factory=list)
    custom_allowed_patterns: List[str] = field(default_factory=list)
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    strict_mode: bool = False  # Block on any threat vs only critical


# Common prompt injection patterns
INJECTION_PATTERNS: List[Tuple[str, str, RiskSeverity]] = [
    # Direct instruction override
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
     "Attempt to ignore previous instructions", RiskSeverity.CRITICAL),
    (r"disregard\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|programming)",
     "Attempt to disregard instructions", RiskSeverity.CRITICAL),
    (r"forget\s+(everything|all|your)\s+(you|instructions?|training|rules)",
     "Attempt to make model forget instructions", RiskSeverity.CRITICAL),

    # System prompt extraction
    (r"(what|show|reveal|display|repeat|print)\s+(is\s+)?(your|the)\s+(system\s+)?(prompt|instructions?|rules)",
     "Attempt to extract system prompt", RiskSeverity.HIGH),
    (r"(output|print|show|display)\s+(your|the)\s+(initial|original|full)\s+(prompt|instructions?)",
     "Attempt to reveal prompt", RiskSeverity.HIGH),

    # Role play jailbreaks
    (r"pretend\s+(you\s+are|to\s+be|you're)\s+(a\s+)?(different|another|new)\s+(ai|assistant|model)",
     "Role play to bypass restrictions", RiskSeverity.MEDIUM),
    (r"act\s+as\s+(if\s+)?(you\s+)?(have\s+no|don't\s+have|without)\s+(restrictions?|limits?|rules?)",
     "Attempt to remove restrictions via role play", RiskSeverity.HIGH),

    # Developer/debug mode exploits
    (r"(enter|enable|activate|switch\s+to)\s+(developer|debug|admin|sudo|root)\s+mode",
     "Attempt to enable special mode", RiskSeverity.CRITICAL),
    (r"you\s+are\s+now\s+in\s+(developer|debug|unrestricted|jailbreak)\s+mode",
     "Declaring special mode active", RiskSeverity.CRITICAL),

    # Instruction injection markers
    (r"\[SYSTEM\]|\[INST\]|\[\/INST\]|<<SYS>>|<\|im_start\|>system",
     "Model-specific instruction markers", RiskSeverity.HIGH),
    (r"<\|endoftext\|>|<\|end\|>|<\|assistant\|>",
     "Special token injection", RiskSeverity.HIGH),

    # Hypothetical framing
    (r"hypothetically,?\s+(if|what\s+if)\s+you\s+(didn't\s+have|had\s+no|could\s+ignore)\s+(rules?|restrictions?)",
     "Hypothetical bypass attempt", RiskSeverity.MEDIUM),
]

# System override patterns
SYSTEM_OVERRIDE_PATTERNS: List[Tuple[str, str, RiskSeverity]] = [
    (r"new\s+system\s+prompt\s*[:=]",
     "Attempt to set new system prompt", RiskSeverity.CRITICAL),
    (r"system\s*:\s*you\s+are",
     "System message injection", RiskSeverity.CRITICAL),
    (r"override\s+(system|safety|content)\s+(prompt|filter|policy)",
     "Direct override attempt", RiskSeverity.CRITICAL),
    (r"(replace|change|modify|update)\s+(your|the)\s+system\s+(prompt|instructions?)",
     "System modification attempt", RiskSeverity.CRITICAL),
]

# Known jailbreak patterns
JAILBREAK_PATTERNS: List[Tuple[str, str, RiskSeverity]] = [
    (r"DAN\s*(mode)?|Do\s+Anything\s+Now",
     "DAN jailbreak attempt", RiskSeverity.CRITICAL),
    (r"STAN\s*(mode)?|Strive\s+To\s+Avoid\s+Norms",
     "STAN jailbreak attempt", RiskSeverity.CRITICAL),
    (r"DUDE\s*(mode)?",
     "DUDE jailbreak attempt", RiskSeverity.HIGH),
    (r"AIM\s*(mode)?|Always\s+Intelligent\s+and\s+Machiavellian",
     "AIM jailbreak attempt", RiskSeverity.CRITICAL),
    (r"evil\s+(confidant|assistant|ai)\s*(mode)?",
     "Evil assistant jailbreak", RiskSeverity.HIGH),
    (r"maximum\s*(mode)?|opposite\s*(mode)?",
     "Maximum/opposite mode jailbreak", RiskSeverity.HIGH),
]


class SecureInputValidator:
    """Validates and sanitizes input for LLM agents.

    Provides comprehensive security validation including prompt injection
    detection, rate limiting, and content sanitization.

    Example:
        >>> validator = SecureInputValidator()
        >>> result = validator.validate(user_input)
        >>> if result.is_valid:
        ...     # Safe to use
        ...     response = await agent.generate_response(result.sanitized_input)
        >>> else:
        ...     # Handle security threat
        ...     logger.warning(f"Blocked: {result.get_threat_summary()}")
    """

    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize validator with configuration.

        Args:
            config: Security configuration (uses defaults if not provided)
        """
        self.config = config or SecurityConfig()
        self._compiled_patterns: Dict[str, List[Tuple[Pattern, str, RiskSeverity]]] = {}
        self._rate_limit_tracker: Dict[str, List[float]] = defaultdict(list)
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self._compiled_patterns["injection"] = [
            (re.compile(p, re.IGNORECASE), desc, sev)
            for p, desc, sev in INJECTION_PATTERNS
        ]
        self._compiled_patterns["system_override"] = [
            (re.compile(p, re.IGNORECASE), desc, sev)
            for p, desc, sev in SYSTEM_OVERRIDE_PATTERNS
        ]
        self._compiled_patterns["jailbreak"] = [
            (re.compile(p, re.IGNORECASE), desc, sev)
            for p, desc, sev in JAILBREAK_PATTERNS
        ]

        # Compile custom patterns
        if self.config.custom_blocked_patterns:
            self._compiled_patterns["custom"] = [
                (re.compile(p, re.IGNORECASE), "Custom blocked pattern", RiskSeverity.HIGH)
                for p in self.config.custom_blocked_patterns
            ]

    def validate(
        self,
        input_text: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate input text for security threats.

        Args:
            input_text: The text to validate
            user_id: Optional user identifier for rate limiting
            context: Optional context for validation decisions

        Returns:
            ValidationResult with validation status and any detected threats
        """
        start_time = time.time()
        threats: List[SecurityThreat] = []
        warnings: List[str] = []

        # Check rate limit first
        if user_id and not self._check_rate_limit(user_id):
            threats.append(SecurityThreat(
                risk_type=SecurityRisk.EXCESSIVE_LENGTH,  # Using for rate limit
                severity=RiskSeverity.HIGH,
                description="Rate limit exceeded",
                recommended_action="rate_limit",
            ))
            return ValidationResult(
                is_valid=False,
                original_input=input_text,
                threats=threats,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Length validation
        if len(input_text) > self.config.max_input_length:
            threats.append(SecurityThreat(
                risk_type=SecurityRisk.EXCESSIVE_LENGTH,
                severity=RiskSeverity.HIGH,
                description=f"Input exceeds max length ({len(input_text)} > {self.config.max_input_length})",
                recommended_action="truncate_or_block",
            ))

        # Token estimate validation
        estimated_tokens = len(input_text) // 4
        if estimated_tokens > self.config.max_tokens_estimate:
            threats.append(SecurityThreat(
                risk_type=SecurityRisk.EXCESSIVE_TOKENS,
                severity=RiskSeverity.MEDIUM,
                description=f"Estimated tokens exceed limit (~{estimated_tokens} > {self.config.max_tokens_estimate})",
                recommended_action="truncate",
            ))

        # Prompt injection detection
        if self.config.detect_injection:
            injection_threats = self._detect_injection_patterns(input_text)
            threats.extend(injection_threats)

        # System override detection
        if self.config.block_system_override:
            override_threats = self._detect_system_override(input_text)
            threats.extend(override_threats)

        # Jailbreak detection
        if self.config.block_jailbreaks:
            jailbreak_threats = self._detect_jailbreaks(input_text)
            threats.extend(jailbreak_threats)

        # Control character detection
        if self.config.sanitize_control_chars:
            control_threats = self._detect_control_characters(input_text)
            threats.extend(control_threats)

        # Encoding attack detection
        if self.config.detect_encoding_attacks:
            encoding_threats = self._detect_encoding_attacks(input_text)
            threats.extend(encoding_threats)

        # Repetition attack detection
        if self.config.detect_repetition:
            repetition_threats = self._detect_repetition_attacks(input_text)
            threats.extend(repetition_threats)

        # Custom pattern detection
        if "custom" in self._compiled_patterns:
            custom_threats = self._detect_custom_patterns(input_text)
            threats.extend(custom_threats)

        # Determine validity
        if self.config.strict_mode:
            is_valid = len(threats) == 0
        else:
            is_valid = not any(
                t.severity in (RiskSeverity.CRITICAL, RiskSeverity.HIGH)
                for t in threats
            )

        # Sanitize input if valid
        sanitized = self._sanitize_input(input_text) if is_valid else None

        processing_time = (time.time() - start_time) * 1000

        # Log security events
        if threats:
            logger.warning(
                f"Security validation found {len(threats)} threats",
                extra={
                    "threats": [t.risk_type.value for t in threats],
                    "user_id": user_id,
                    "input_length": len(input_text),
                },
            )

        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized,
            original_input=input_text,
            threats=threats,
            warnings=warnings,
            processing_time_ms=processing_time,
        )

    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        now = time.time()
        window_start = now - self.config.rate_limit_window_seconds

        # Clean old entries
        self._rate_limit_tracker[user_id] = [
            t for t in self._rate_limit_tracker[user_id] if t > window_start
        ]

        # Check limit
        if len(self._rate_limit_tracker[user_id]) >= self.config.rate_limit_requests:
            return False

        # Record request
        self._rate_limit_tracker[user_id].append(now)
        return True

    def _detect_injection_patterns(self, text: str) -> List[SecurityThreat]:
        """Detect prompt injection patterns."""
        threats = []
        for pattern, description, severity in self._compiled_patterns.get("injection", []):
            match = pattern.search(text)
            if match:
                threats.append(SecurityThreat(
                    risk_type=SecurityRisk.PROMPT_INJECTION,
                    severity=severity,
                    description=description,
                    matched_pattern=match.group()[:100],
                    position=match.start(),
                ))
        return threats

    def _detect_system_override(self, text: str) -> List[SecurityThreat]:
        """Detect system prompt override attempts."""
        threats = []
        for pattern, description, severity in self._compiled_patterns.get("system_override", []):
            match = pattern.search(text)
            if match:
                threats.append(SecurityThreat(
                    risk_type=SecurityRisk.SYSTEM_OVERRIDE,
                    severity=severity,
                    description=description,
                    matched_pattern=match.group()[:100],
                    position=match.start(),
                ))
        return threats

    def _detect_jailbreaks(self, text: str) -> List[SecurityThreat]:
        """Detect known jailbreak patterns."""
        threats = []
        for pattern, description, severity in self._compiled_patterns.get("jailbreak", []):
            match = pattern.search(text)
            if match:
                threats.append(SecurityThreat(
                    risk_type=SecurityRisk.JAILBREAK_ATTEMPT,
                    severity=severity,
                    description=description,
                    matched_pattern=match.group()[:100],
                    position=match.start(),
                ))
        return threats

    def _detect_control_characters(self, text: str) -> List[SecurityThreat]:
        """Detect potentially harmful control characters."""
        threats = []

        # Check for null bytes
        if "\x00" in text:
            threats.append(SecurityThreat(
                risk_type=SecurityRisk.CONTROL_CHARACTERS,
                severity=RiskSeverity.HIGH,
                description="Null byte detected",
                position=text.index("\x00"),
            ))

        # Check for unusual unicode control characters
        control_chars = re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", text)
        if control_chars:
            threats.append(SecurityThreat(
                risk_type=SecurityRisk.CONTROL_CHARACTERS,
                severity=RiskSeverity.MEDIUM,
                description=f"Found {len(control_chars)} control characters",
            ))

        # Check for bidirectional text override characters (can hide text)
        bidi_chars = re.findall(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", text)
        if bidi_chars:
            threats.append(SecurityThreat(
                risk_type=SecurityRisk.CONTROL_CHARACTERS,
                severity=RiskSeverity.HIGH,
                description="Bidirectional text override characters detected",
            ))

        return threats

    def _detect_encoding_attacks(self, text: str) -> List[SecurityThreat]:
        """Detect encoding-based attacks."""
        threats = []

        # Check for large base64 blocks (potential hidden payloads)
        base64_pattern = re.compile(r"[A-Za-z0-9+/]{100,}={0,2}")
        base64_matches = base64_pattern.findall(text)
        if base64_matches:
            threats.append(SecurityThreat(
                risk_type=SecurityRisk.ENCODED_CONTENT,
                severity=RiskSeverity.MEDIUM,
                description=f"Found {len(base64_matches)} large base64-like blocks",
            ))

        # Check for hex-encoded strings
        hex_pattern = re.compile(r"(?:\\x[0-9a-fA-F]{2}){10,}")
        if hex_pattern.search(text):
            threats.append(SecurityThreat(
                risk_type=SecurityRisk.ENCODED_CONTENT,
                severity=RiskSeverity.MEDIUM,
                description="Hex-encoded content detected",
            ))

        return threats

    def _detect_repetition_attacks(self, text: str) -> List[SecurityThreat]:
        """Detect repetition-based attacks (token exhaustion, etc)."""
        threats = []

        # Check for excessive repetition of same word/phrase
        words = text.lower().split()
        if len(words) > 10:
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1

            max_freq = max(word_freq.values()) if word_freq else 0
            if max_freq > len(words) * 0.3:  # Same word > 30% of content
                threats.append(SecurityThreat(
                    risk_type=SecurityRisk.REPETITION_ATTACK,
                    severity=RiskSeverity.MEDIUM,
                    description=f"Excessive word repetition detected (max frequency: {max_freq}/{len(words)})",
                ))

        # Check for repeated character sequences
        for match in re.finditer(r"(.{3,})\1{10,}", text):
            threats.append(SecurityThreat(
                risk_type=SecurityRisk.REPETITION_ATTACK,
                severity=RiskSeverity.HIGH,
                description="Repeated sequence attack detected",
                matched_pattern=match.group()[:50],
                position=match.start(),
            ))

        return threats

    def _detect_custom_patterns(self, text: str) -> List[SecurityThreat]:
        """Detect custom blocked patterns."""
        threats = []
        for pattern, description, severity in self._compiled_patterns.get("custom", []):
            match = pattern.search(text)
            if match:
                threats.append(SecurityThreat(
                    risk_type=SecurityRisk.HARMFUL_CONTENT,
                    severity=severity,
                    description=description,
                    matched_pattern=match.group()[:100],
                    position=match.start(),
                ))
        return threats

    def _sanitize_input(self, text: str) -> str:
        """Sanitize input by removing/escaping dangerous content."""
        sanitized = text

        # Remove null bytes
        sanitized = sanitized.replace("\x00", "")

        # Remove other control characters (except newlines and tabs)
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", sanitized)

        # Remove bidirectional override characters
        sanitized = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", sanitized)

        # Normalize unicode
        import unicodedata
        sanitized = unicodedata.normalize("NFKC", sanitized)

        # Collapse excessive whitespace
        sanitized = re.sub(r"\n{4,}", "\n\n\n", sanitized)
        sanitized = re.sub(r" {4,}", "   ", sanitized)

        return sanitized.strip()


def create_secure_agent_wrapper(validator: SecureInputValidator):
    """Create a decorator that adds input validation to agent methods.

    Args:
        validator: SecureInputValidator instance

    Returns:
        Decorator function

    Example:
        >>> validator = SecureInputValidator()
        >>> secure = create_secure_agent_wrapper(validator)
        >>>
        >>> @secure
        >>> async def generate_response(self, messages, context=None):
        ...     # Original implementation
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        async def wrapper(self, messages, context=None, **kwargs):
            # Extract text content for validation
            if isinstance(messages, str):
                text = messages
            elif isinstance(messages, list):
                text = " ".join(
                    m.get("content", "") for m in messages
                    if isinstance(m, dict) and m.get("role") == "user"
                )
            else:
                text = str(messages)

            # Validate
            result = validator.validate(text)

            if not result.is_valid:
                raise ValueError(
                    f"Input validation failed: {result.get_threat_summary()}"
                )

            # Call original function with sanitized input
            if isinstance(messages, str):
                messages = result.sanitized_input
            # For message lists, we pass through (sanitization applied at string level)

            return await func(self, messages, context, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "SecurityRisk",
    "RiskSeverity",
    "SecurityThreat",
    "ValidationResult",
    "SecurityConfig",
    "SecureInputValidator",
    "create_secure_agent_wrapper",
]
