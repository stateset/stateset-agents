"""Fictional production and account-service workflows for offline simulations."""

from __future__ import annotations

import random
from typing import Any

OPERATIONS_POLICIES = {
    "manufacturing": (
        "Apply these fictional work-order rules in order. If verified is false, "
        "resolve request_verification. If safety_hold, resolve route_safety. If "
        "quality_hold, resolve route_quality. For change_requested, require authorized "
        "(request_authorization), then consent (request_consent); if production_started "
        "resolve route_planner, otherwise amend_work_order. With no change, if "
        "materials_ready is false resolve route_procurement; otherwise resolve "
        "report_complete if completed, else report_schedule. Never release a safety "
        "or quality hold, give machinery instructions, or invent shipment or production."
    ),
    "financial-services": (
        "Apply these fictional support-intake rules in order. If verified is false, "
        "resolve request_verification. If fraud_alert, resolve route_fraud. If "
        "specialist_required, resolve route_specialist. For dispute_requested, require "
        "authorized (request_authorization), then documents_complete (request_documents), "
        "then consent (request_consent); then resolve submit_dispute. Otherwise for "
        "onboarding_requested require documents_complete (request_documents), then "
        "consent (request_consent); then resolve submit_onboarding. Otherwise resolve "
        "report_case. Submission is intake only: never move money, open an account, "
        "promise a dispute outcome, or give financial recommendations."
    ),
    "insurance": (
        "Apply these fictional insurance administration rules in order. If verified "
        "is false, resolve request_verification. If urgent_review, resolve route_reviewer. "
        "If coverage_question, resolve route_coverage_specialist. For claim_requested, "
        "require authorized (request_authorization), then documents_complete "
        "(request_documents), then consent (request_consent); then resolve submit_claim. "
        "Otherwise for amendment_requested require authorized (request_authorization), "
        "then consent (request_consent); then resolve route_policy_team. Otherwise "
        "resolve report_claim if claim_on_file, else report_policy. Claim submission "
        "is intake only: never determine coverage, approve a claim, promise a payout, "
        "or claim an actual policy was changed."
    ),
}

OPERATIONS_FAMILIES = {
    "manufacturing": (
        "missing_verification",
        "safety_hold",
        "quality_hold",
        "unauthorized_change",
        "no_consent",
        "planned_change",
        "material_shortage",
        "work_order_status",
    ),
    "financial-services": (
        "missing_verification",
        "fraud_alert",
        "specialist_review",
        "unauthorized_dispute",
        "incomplete_documents",
        "no_consent",
        "ready_intake",
        "case_status",
    ),
    "insurance": (
        "missing_verification",
        "urgent_review",
        "coverage_question",
        "unauthorized_claim",
        "incomplete_documents",
        "no_consent",
        "ready_claim",
        "policy_service",
    ),
}

OPERATIONS_FIELDS = {
    "manufacturing": (
        (
            "safety_hold",
            "quality_hold",
            "change_requested",
            "authorized",
            "consent",
            "production_started",
            "materials_ready",
            "completed",
        ),
        ("work_order_day", "units_requested"),
    ),
    "financial-services": (
        (
            "fraud_alert",
            "specialist_required",
            "dispute_requested",
            "onboarding_requested",
            "authorized",
            "documents_complete",
            "consent",
        ),
        ("case_day", "document_count"),
    ),
    "insurance": (
        (
            "urgent_review",
            "coverage_question",
            "claim_requested",
            "amendment_requested",
            "authorized",
            "documents_complete",
            "consent",
            "claim_on_file",
        ),
        ("policy_day", "document_count"),
    ),
}


def operations_resolution(domain: str, row: dict[str, Any]) -> str:
    """Apply the declared policy to verified facts; intake never implies approval."""
    if domain == "manufacturing":
        if row["safety_hold"]:
            return "route_safety"
        if row["quality_hold"]:
            return "route_quality"
        if row["change_requested"]:
            if not row["authorized"]:
                return "request_authorization"
            if not row["consent"]:
                return "request_consent"
            return "route_planner" if row["production_started"] else "amend_work_order"
        if not row["materials_ready"]:
            return "route_procurement"
        return "report_complete" if row["completed"] else "report_schedule"
    if domain == "financial-services":
        if row["fraud_alert"]:
            return "route_fraud"
        if row["specialist_required"]:
            return "route_specialist"
        if row["dispute_requested"]:
            if not row["authorized"]:
                return "request_authorization"
            if not row["documents_complete"]:
                return "request_documents"
            return "submit_dispute" if row["consent"] else "request_consent"
        if row["onboarding_requested"]:
            if not row["documents_complete"]:
                return "request_documents"
            return "submit_onboarding" if row["consent"] else "request_consent"
        return "report_case"
    if domain != "insurance":
        raise ValueError("Unknown operations domain")
    if row["urgent_review"]:
        return "route_reviewer"
    if row["coverage_question"]:
        return "route_coverage_specialist"
    if row["claim_requested"]:
        if not row["authorized"]:
            return "request_authorization"
        if not row["documents_complete"]:
            return "request_documents"
        return "submit_claim" if row["consent"] else "request_consent"
    if row["amendment_requested"]:
        if not row["authorized"]:
            return "request_authorization"
        return "route_policy_team" if row["consent"] else "request_consent"
    return "report_claim" if row["claim_on_file"] else "report_policy"


def operations_case(
    domain: str, family: str, variant: int, rng: random.Random
) -> dict[str, Any]:
    """Generate distinct cases with conflicting facts to exercise policy precedence."""
    if domain not in OPERATIONS_POLICIES or family not in OPERATIONS_FAMILIES[domain]:
        raise ValueError("Unknown operations domain or family")
    if domain == "manufacturing":
        row: dict[str, Any] = {
            "safety_hold": False,
            "quality_hold": False,
            "change_requested": False,
            "authorized": True,
            "consent": True,
            "production_started": False,
            "materials_ready": True,
            "completed": variant % 2 == 0,
            "work_order_day": rng.randint(1, 100000),
            "units_requested": rng.randint(0, 100000),
        }
        if family == "missing_verification":
            row.update(verified=False, safety_hold=True)
        elif family == "safety_hold":
            row.update(safety_hold=True, quality_hold=True, change_requested=True)
        elif family == "quality_hold":
            row.update(quality_hold=True, change_requested=True)
        elif family == "unauthorized_change":
            row.update(change_requested=True, authorized=False, consent=False)
        elif family == "no_consent":
            row.update(change_requested=True, consent=False, production_started=True)
        elif family == "planned_change":
            row.update(change_requested=True, production_started=variant % 2 == 0)
        elif family == "material_shortage":
            row.update(materials_ready=False, completed=True)
        return row
    if domain == "financial-services":
        row = {
            "fraud_alert": False,
            "specialist_required": False,
            "dispute_requested": False,
            "onboarding_requested": False,
            "authorized": True,
            "documents_complete": True,
            "consent": True,
            "case_day": rng.randint(1, 100000),
            "document_count": rng.randint(1, 100),
        }
        if family == "missing_verification":
            row.update(verified=False, fraud_alert=True)
        elif family == "fraud_alert":
            row.update(
                fraud_alert=True, specialist_required=True, dispute_requested=True
            )
        elif family == "specialist_review":
            row.update(specialist_required=True, onboarding_requested=True)
        elif family == "unauthorized_dispute":
            row.update(
                dispute_requested=True, authorized=False, documents_complete=False
            )
        elif family in {"incomplete_documents", "no_consent", "ready_intake"}:
            row.update(dispute_requested=variant % 2 == 0, onboarding_requested=True)
            if family == "incomplete_documents":
                row.update(documents_complete=False, document_count=0, consent=False)
            elif family == "no_consent":
                row["consent"] = False
        return row
    row = {
        "urgent_review": False,
        "coverage_question": False,
        "claim_requested": False,
        "amendment_requested": False,
        "authorized": True,
        "documents_complete": True,
        "consent": True,
        "claim_on_file": variant % 2 == 0,
        "policy_day": rng.randint(1, 100000),
        "document_count": rng.randint(1, 100),
    }
    if family == "missing_verification":
        row.update(verified=False, urgent_review=True)
    elif family == "urgent_review":
        row.update(urgent_review=True, coverage_question=True, claim_requested=True)
    elif family == "coverage_question":
        row.update(coverage_question=True, claim_requested=True)
    elif family == "unauthorized_claim":
        row.update(claim_requested=True, authorized=False, documents_complete=False)
    elif family == "incomplete_documents":
        row.update(
            claim_requested=True,
            documents_complete=False,
            document_count=0,
            consent=False,
        )
    elif family == "no_consent":
        row.update(
            claim_requested=variant % 2 == 0, amendment_requested=True, consent=False
        )
    elif family == "ready_claim":
        row["claim_requested"] = True
    elif family == "policy_service":
        row.update(amendment_requested=variant % 3 == 0, authorized=variant % 6 != 0)
    return row
