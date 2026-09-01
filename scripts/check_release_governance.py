#!/usr/bin/env python3
"""Validate component maturity and security-response release governance."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATURITY_MANIFEST = REPO_ROOT / "contracts" / "component_maturity_v1.json"
DEFAULT_SECURITY_MANIFEST = REPO_ROOT / "contracts" / "security_response_v1.json"
DEFAULT_API_MANIFEST = REPO_ROOT / "contracts" / "public_api_v1.json"
DEFAULT_MATURITY_DOC = REPO_ROOT / "docs" / "COMPONENT_MATURITY.md"
DEFAULT_SECURITY_DOC = REPO_ROOT / "SECURITY.md"

REQUIRED_COMPONENT_IDS = {
    "api-serving",
    "core-agents",
    "deployment-assets",
    "distributed-rollouts",
    "environments-rewards",
    "external-training-backends",
    "gspo-dapo-gepo",
    "grpo-training",
    "memory",
    "offline-rl",
    "provider-control-plane",
    "research-modules",
    "reward-learning-rlaif",
    "rust-acceleration",
    "sim-to-real",
    "vapo",
}
MATURITY_LEVELS = {"stable", "beta", "experimental"}
OWNERS = {"api", "core", "performance", "platform", "research", "training"}
COMPONENT_KEYS = {
    "docs",
    "evidence",
    "graduation_criteria",
    "id",
    "limitations",
    "maturity",
    "name",
    "owner",
    "public_contract_refs",
    "tests",
}


class GovernanceError(RuntimeError):
    """Raised when governance inputs cannot be loaded safely."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GovernanceError(f"manifest does not exist: {path}") from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise GovernanceError(f"manifest is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GovernanceError(f"manifest must contain a JSON object: {path}")
    return value


def _strings(value: Any, field: str, errors: list[str]) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        errors.append(f"{field} must be a list of non-empty strings")
        return []
    if len(value) != len(set(value)):
        errors.append(f"{field} contains duplicates")
    return value


def _validate_repo_path(value: str, field: str, errors: list[str]) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        errors.append(f"{field} must be a repository-relative path: {value}")
        return
    candidate = REPO_ROOT / path
    if not candidate.is_file():
        errors.append(f"{field} does not reference a file: {value}")


def _public_contract_refs(api_manifest: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    python_contract = api_manifest.get("python", {})
    if isinstance(python_contract, dict):
        for module, exports in python_contract.items():
            if isinstance(module, str) and isinstance(exports, dict):
                refs.update(f"python:{module}.{name}" for name in exports)
    operations = api_manifest.get("http", {}).get("operations", {})
    if isinstance(operations, dict):
        refs.update(f"http:{operation}" for operation in operations)
    return refs


def validate_maturity(
    manifest: dict[str, Any],
    api_manifest: dict[str, Any],
    documentation: str,
) -> list[str]:
    """Return component-maturity policy violations."""
    errors: list[str] = []
    if set(manifest) != {"components", "levels", "schema_version"}:
        errors.append("maturity manifest has unexpected or missing top-level keys")
    if manifest.get("schema_version") != 1:
        errors.append("maturity schema_version must be 1")
    levels = manifest.get("levels")
    if not isinstance(levels, dict) or set(levels) != MATURITY_LEVELS:
        errors.append("maturity levels must define stable, beta, and experimental")

    components = manifest.get("components")
    if not isinstance(components, list):
        return errors + ["components must be a list"]

    ids: list[str] = []
    known_contract_refs = _public_contract_refs(api_manifest)
    for index, component in enumerate(components):
        prefix = f"components[{index}]"
        if not isinstance(component, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if set(component) != COMPONENT_KEYS:
            errors.append(f"{prefix} has unexpected or missing keys")

        component_id = component.get("id")
        if not isinstance(component_id, str) or not re.fullmatch(
            r"[a-z0-9]+(?:-[a-z0-9]+)*", component_id
        ):
            errors.append(f"{prefix}.id must be lowercase kebab-case")
            component_id = f"invalid-{index}"
        ids.append(component_id)

        for field in ("name", "owner", "maturity"):
            if (
                not isinstance(component.get(field), str)
                or not component[field].strip()
            ):
                errors.append(f"{prefix}.{field} must be a non-empty string")
        maturity = component.get("maturity")
        if maturity not in MATURITY_LEVELS:
            errors.append(f"{prefix}.maturity is invalid: {maturity!r}")
        if component.get("owner") not in OWNERS:
            errors.append(f"{prefix}.owner is not a recognized accountable team")

        docs = _strings(component.get("docs"), f"{prefix}.docs", errors)
        tests = _strings(component.get("tests"), f"{prefix}.tests", errors)
        evidence = _strings(component.get("evidence"), f"{prefix}.evidence", errors)
        limitations = _strings(
            component.get("limitations"), f"{prefix}.limitations", errors
        )
        graduation = _strings(
            component.get("graduation_criteria"),
            f"{prefix}.graduation_criteria",
            errors,
        )
        contract_refs = _strings(
            component.get("public_contract_refs"),
            f"{prefix}.public_contract_refs",
            errors,
        )

        for field, values in (("docs", docs), ("tests", tests), ("evidence", evidence)):
            if not values:
                errors.append(f"{prefix}.{field} must not be empty")
            for value in values:
                _validate_repo_path(value, f"{prefix}.{field}", errors)
        for test_path in tests:
            if not test_path.startswith("tests/") or not Path(
                test_path
            ).name.startswith("test_"):
                errors.append(f"{prefix}.tests is not a pytest file: {test_path}")

        unknown_refs = sorted(set(contract_refs) - known_contract_refs)
        if unknown_refs:
            errors.append(f"{prefix} has unknown public contract refs: {unknown_refs}")
        if maturity == "stable":
            if not contract_refs:
                errors.append(
                    f"{prefix} stable components require public contract refs"
                )
            if limitations or graduation:
                errors.append(
                    f"{prefix} stable components cannot have limitations or graduation criteria"
                )
        elif maturity in {"beta", "experimental"}:
            if not limitations:
                errors.append(f"{prefix} {maturity} components require limitations")
            if not graduation:
                errors.append(
                    f"{prefix} {maturity} components require graduation criteria"
                )

        display_level = str(maturity).capitalize()
        if f"| `{component_id}` | {display_level} |" not in documentation:
            errors.append(
                f"maturity documentation is missing {component_id}: {display_level}"
            )

    if len(ids) != len(set(ids)):
        errors.append("component ids must be unique")
    missing_ids = sorted(REQUIRED_COMPONENT_IDS - set(ids))
    extra_ids = sorted(set(ids) - REQUIRED_COMPONENT_IDS)
    if missing_ids or extra_ids:
        errors.append(
            f"component inventory differs: missing={missing_ids}, extra={extra_ids}"
        )
    return errors


def validate_security(manifest: dict[str, Any], documentation: str) -> list[str]:
    """Return security-response policy violations."""
    errors: list[str] = []
    required_keys = {
        "coordinated_disclosure_max_days",
        "reporting_channels",
        "safe_harbor",
        "schema_version",
        "service_levels",
        "supported_versions",
        "third_party_review",
    }
    if set(manifest) != required_keys:
        errors.append("security manifest has unexpected or missing top-level keys")
    if manifest.get("schema_version") != 1:
        errors.append("security schema_version must be 1")
    if manifest.get("safe_harbor") is not True:
        errors.append("security policy must publish safe harbor")
    if manifest.get("coordinated_disclosure_max_days") != 90:
        errors.append("coordinated disclosure maximum must be 90 days")

    channels = manifest.get("reporting_channels")
    expected_channels = {
        "mailto:security@stateset.ai",
        "github-private-vulnerability-reporting",
    }
    if not isinstance(channels, list) or set(channels) != expected_channels:
        errors.append("security reporting channels are incomplete")

    supported = manifest.get("supported_versions")
    if supported != {"latest_minor": True, "older_minors": False}:
        errors.append("supported-version policy must cover only the latest minor")

    service_levels = manifest.get("service_levels")
    expected_slas = {
        "acknowledgement_hours": 24,
        "remediation_target_calendar_days": {
            "critical": 7,
            "high": 14,
            "low": 90,
            "medium": 30,
        },
        "status_update_hours_critical_high": 72,
        "triage_hours": 72,
    }
    if service_levels != expected_slas:
        errors.append("security service levels differ from the reviewed v1 policy")

    review = manifest.get("third_party_review")
    if not isinstance(review, dict) or set(review) != {"evidence", "status"}:
        errors.append("third_party_review must contain status and evidence")
    else:
        status = review.get("status")
        evidence = review.get("evidence")
        if status not in {"pending", "complete"}:
            errors.append("third_party_review status must be pending or complete")
        evidence_paths = _strings(evidence, "third_party_review.evidence", errors)
        for value in evidence_paths:
            _validate_repo_path(value, "third_party_review.evidence", errors)
        if status == "complete" and not evidence_paths:
            errors.append("completed third-party review requires retained evidence")
        if status == "pending" and evidence_paths:
            errors.append("pending third-party review cannot claim evidence")

    required_doc_fragments = {
        "security@stateset.ai",
        "GitHub private vulnerability reporting",
        "| Acknowledge report | 24 hours |",
        "| Complete initial severity triage | 72 hours |",
        "| Critical remediation target | 7 calendar days |",
        "| High remediation target | 14 calendar days |",
        "| Medium remediation target | 30 calendar days |",
        "| Low remediation target | 90 calendar days |",
        "embargo target of 90 days",
        "## Safe harbor",
        "independent third-party security",
    }
    for fragment in sorted(required_doc_fragments):
        if fragment not in documentation:
            errors.append(f"security documentation is missing: {fragment}")
    return errors


def validate_governance(
    maturity_manifest_path: Path = DEFAULT_MATURITY_MANIFEST,
    security_manifest_path: Path = DEFAULT_SECURITY_MANIFEST,
    api_manifest_path: Path = DEFAULT_API_MANIFEST,
    maturity_doc_path: Path = DEFAULT_MATURITY_DOC,
    security_doc_path: Path = DEFAULT_SECURITY_DOC,
) -> list[str]:
    """Load and validate every release-governance contract."""
    maturity_manifest = _load_object(maturity_manifest_path)
    security_manifest = _load_object(security_manifest_path)
    api_manifest = _load_object(api_manifest_path)
    try:
        maturity_doc = maturity_doc_path.read_text(encoding="utf-8")
        security_doc = security_doc_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        raise GovernanceError(f"governance documentation is unreadable: {exc}") from exc
    return validate_maturity(
        maturity_manifest, api_manifest, maturity_doc
    ) + validate_security(security_manifest, security_doc)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maturity-manifest", type=Path, default=DEFAULT_MATURITY_MANIFEST
    )
    parser.add_argument(
        "--security-manifest", type=Path, default=DEFAULT_SECURITY_MANIFEST
    )
    parser.add_argument("--api-manifest", type=Path, default=DEFAULT_API_MANIFEST)
    parser.add_argument("--maturity-doc", type=Path, default=DEFAULT_MATURITY_DOC)
    parser.add_argument("--security-doc", type=Path, default=DEFAULT_SECURITY_DOC)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        errors = validate_governance(
            args.maturity_manifest,
            args.security_manifest,
            args.api_manifest,
            args.maturity_doc,
            args.security_doc,
        )
    except GovernanceError as exc:
        print(f"Release governance check failed: {exc}", file=sys.stderr)
        return 1
    if errors:
        print("Release governance check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Release governance check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
