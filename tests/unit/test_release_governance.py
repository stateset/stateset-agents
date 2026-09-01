"""Tests for evidence-backed component and security governance."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.check_release_governance import (
    DEFAULT_API_MANIFEST,
    DEFAULT_MATURITY_DOC,
    DEFAULT_MATURITY_MANIFEST,
    DEFAULT_SECURITY_DOC,
    DEFAULT_SECURITY_MANIFEST,
    REQUIRED_COMPONENT_IDS,
    validate_governance,
    validate_maturity,
    validate_security,
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_release_governance_is_valid() -> None:
    assert validate_governance() == []


def test_maturity_manifest_covers_canonical_product_domains() -> None:
    maturity = _load(DEFAULT_MATURITY_MANIFEST)
    components = maturity["components"]
    assert isinstance(components, list)
    assert {component["id"] for component in components} == REQUIRED_COMPONENT_IDS
    assert {component["maturity"] for component in components} == {
        "stable",
        "beta",
        "experimental",
    }


def test_stable_component_requires_valid_contract_references() -> None:
    maturity = _load(DEFAULT_MATURITY_MANIFEST)
    api = _load(DEFAULT_API_MANIFEST)
    documentation = DEFAULT_MATURITY_DOC.read_text(encoding="utf-8")
    mutated = copy.deepcopy(maturity)
    component = mutated["components"][0]
    component["public_contract_refs"] = ["python:stateset_agents.DoesNotExist"]

    errors = validate_maturity(mutated, api, documentation)
    assert any("unknown public contract refs" in error for error in errors)


def test_nonstable_component_requires_limitations_and_graduation_criteria() -> None:
    maturity = _load(DEFAULT_MATURITY_MANIFEST)
    api = _load(DEFAULT_API_MANIFEST)
    documentation = DEFAULT_MATURITY_DOC.read_text(encoding="utf-8")
    mutated = copy.deepcopy(maturity)
    component = next(
        item for item in mutated["components"] if item["maturity"] == "beta"
    )
    component["limitations"] = []
    component["graduation_criteria"] = []

    errors = validate_maturity(mutated, api, documentation)
    assert any("beta components require limitations" in error for error in errors)
    assert any(
        "beta components require graduation criteria" in error for error in errors
    )


def test_maturity_evidence_paths_must_exist() -> None:
    maturity = _load(DEFAULT_MATURITY_MANIFEST)
    api = _load(DEFAULT_API_MANIFEST)
    documentation = DEFAULT_MATURITY_DOC.read_text(encoding="utf-8")
    mutated = copy.deepcopy(maturity)
    mutated["components"][0]["evidence"] = ["benchmark_results/not-real.json"]

    errors = validate_maturity(mutated, api, documentation)
    assert any("does not reference a file" in error for error in errors)


def test_security_policy_rejects_weakened_sla() -> None:
    security = _load(DEFAULT_SECURITY_MANIFEST)
    documentation = DEFAULT_SECURITY_DOC.read_text(encoding="utf-8")
    mutated = copy.deepcopy(security)
    mutated["service_levels"]["acknowledgement_hours"] = 168

    errors = validate_security(mutated, documentation)
    assert "security service levels differ from the reviewed v1 policy" in errors


def test_third_party_review_cannot_be_claimed_without_evidence() -> None:
    security = _load(DEFAULT_SECURITY_MANIFEST)
    documentation = DEFAULT_SECURITY_DOC.read_text(encoding="utf-8")
    mutated = copy.deepcopy(security)
    mutated["third_party_review"]["status"] = "complete"

    errors = validate_security(mutated, documentation)
    assert "completed third-party review requires retained evidence" in errors


def test_ci_runs_release_governance_gate() -> None:
    workflow = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    assert "Verify release governance" in workflow
    assert "make release-governance" in workflow
