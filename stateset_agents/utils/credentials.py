"""Non-secret identifiers for high-entropy credentials.

These fingerprints are for logging, ownership, and rate-limit bucketing. They
are not password hashes and must never be used to recover or authenticate a
credential.
"""

from __future__ import annotations

import hmac

_FINGERPRINT_DOMAIN = b"stateset-cred-v1"


def credential_fingerprint(value: str) -> str:
    """Return a stable, domain-separated opaque credential identifier."""
    return hmac.digest(_FINGERPRINT_DOMAIN, value.encode("utf-8"), "sha256").hex()[:16]
