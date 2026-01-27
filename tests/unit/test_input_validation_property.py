import re

from hypothesis import given, settings, strategies as st

from stateset_agents.core.input_validation import SecurityConfig, SecureInputValidator

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
BIDI_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")

TEXT_STRATEGY = st.text(
    alphabet=st.characters(blacklist_categories=["Cs"]),
    max_size=200,
)


@settings(max_examples=75)
@given(TEXT_STRATEGY)
def test_sanitize_removes_control_and_bidi_chars(text: str) -> None:
    validator = SecureInputValidator(SecurityConfig())
    sanitized = validator._sanitize_input(text)

    assert "\x00" not in sanitized
    assert CONTROL_RE.search(sanitized) is None
    assert BIDI_RE.search(sanitized) is None
    assert sanitized == sanitized.strip()


@settings(max_examples=75)
@given(TEXT_STRATEGY)
def test_validate_returns_sanitized_input_for_any_text(text: str) -> None:
    config = SecurityConfig(
        max_input_length=10000,
        max_tokens_estimate=10000,
        detect_injection=False,
        block_system_override=False,
        block_jailbreaks=False,
        sanitize_control_chars=False,
        detect_encoding_attacks=False,
        detect_repetition=False,
    )
    validator = SecureInputValidator(config)
    result = validator.validate(text)

    assert result.is_valid
    assert result.sanitized_input is not None
    assert CONTROL_RE.search(result.sanitized_input) is None
    assert BIDI_RE.search(result.sanitized_input) is None
