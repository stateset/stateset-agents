"""Narrow, time-limited exceptions for the release security gate."""

from datetime import date
from typing import Any

# Approved for v0.56.0: upstream closed the proposed fixes as outside its
# security policy. Reassess before this date or when a patched release appears.
ACCELERATE_EXCEPTION_REVIEW_DATE = date(2026, 12, 31)


def is_approved_pip_audit_exception(
    package: Any, version: Any, vulnerability: Any, *, today: date | None = None
) -> bool:
    """Recognize only the reviewed Accelerate advisory in pip-audit output."""
    if today is None:
        today = date.today()
    if not isinstance(vulnerability, dict):
        return False
    aliases = vulnerability.get("aliases")
    return (
        today <= ACCELERATE_EXCEPTION_REVIEW_DATE
        and package == "accelerate"
        and version == "1.14.0"
        and vulnerability.get("id") == "PYSEC-2026-3804"
        and isinstance(aliases, list)
        and "CVE-2026-69112" in aliases
        and "GHSA-4j2p-28q2-5m79" in aliases
    )


def classify_pip_audit_findings(
    payload: Any, *, today: date | None = None
) -> tuple[list[str], list[str]]:
    """Validate a complete audit report and partition its findings."""
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("dependencies"), list)
        or not payload["dependencies"]
    ):
        raise ValueError("pip-audit report has no dependency list")
    approved: list[str] = []
    blocked: list[str] = []
    for dependency in payload["dependencies"]:
        if not isinstance(dependency, dict):
            raise ValueError("invalid pip-audit dependency entry")
        name = dependency.get("name")
        version = dependency.get("version")
        vulns = dependency.get("vulns")
        if (
            not isinstance(name, str)
            or not isinstance(version, str)
            or not isinstance(vulns, list)
        ):
            raise ValueError("incomplete pip-audit dependency entry")
        for vulnerability in vulns:
            if not isinstance(vulnerability, dict) or not isinstance(
                vulnerability.get("id"), str
            ):
                raise ValueError("invalid pip-audit vulnerability entry")
            finding = f"{name} {version}: {vulnerability['id']}"
            if is_approved_pip_audit_exception(
                name, version, vulnerability, today=today
            ):
                approved.append(finding)
            else:
                blocked.append(finding)
    return approved, blocked
