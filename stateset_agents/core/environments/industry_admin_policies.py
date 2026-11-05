"""Fictional administrative workflows; no clinical or eligibility decisions."""

from __future__ import annotations

import random
from typing import Any

ADMIN_POLICIES = {
    "media": (
        "Apply these fictional subscription rules in order. If verified is false, "
        "resolve request_verification. If security_alert, resolve escalate_security. "
        "If service_outage, resolve report_outage. For cancel_requested or "
        "change_requested, require authorized (request_authorization), then consent "
        "(request_consent). Cancel resolves cancel_subscription without requiring "
        "a quote; a change requires price_confirmed (request_quote), then resolves "
        "change_subscription. Otherwise if entitled is false, resolve route_access; "
        "if balance_due_cents > 0, resolve route_billing; otherwise report_subscription. "
        "Do not invent charges, restored access, or an outage recovery time."
    ),
    "healthcare": (
        "Apply these fictional appointment administration rules in order. If "
        "verified is false, resolve request_verification. If staff_review, resolve "
        "route_staff. For cancel_requested, reschedule_requested, or booking_requested, "
        "require authorized (request_authorization), then consent (request_consent). "
        "Cancel resolves cancel_appointment without a slot check. Otherwise require "
        "slot_available (offer_alternatives); then resolve reschedule_appointment "
        "for reschedule_requested, otherwise book_appointment. With no requested "
        "change resolve report_appointment. This is administration only: do not "
        "diagnose, triage, give treatment advice, or claim care was delivered."
    ),
    "public-sector": (
        "Apply these fictional application administration rules in order. If verified "
        "is false, resolve request_verification. If requires_office_review, resolve "
        "route_office. For submission_requested, require authorized "
        "(request_authorization), then documents_complete (request_documents), then "
        "consent (request_consent); then resolve submit_application. Otherwise for "
        "appointment_requested, require authorized (request_authorization), then "
        "consent (request_consent), then slot_available (offer_alternatives); then "
        "resolve book_appointment. Otherwise resolve report_submitted if submitted, "
        "else report_draft. Submission is not approval. Never decide eligibility, "
        "invent an approval, or promise a real agency processing time."
    ),
}

ADMIN_FAMILIES = {
    "media": (
        "missing_verification",
        "security_incident",
        "outage",
        "cancellation",
        "subscription_change",
        "access",
        "billing",
        "subscription_status",
    ),
    "healthcare": (
        "missing_verification",
        "staff_review",
        "unauthorized_change",
        "no_consent",
        "no_slot",
        "booking",
        "reschedule",
        "cancellation",
    ),
    "public-sector": (
        "missing_verification",
        "office_review",
        "incomplete_documents",
        "unauthorized_submission",
        "unconfirmed_submission",
        "ready_submission",
        "appointment",
        "case_status",
    ),
}

ADMIN_FIELDS = {
    "media": (
        (
            "security_alert",
            "service_outage",
            "cancel_requested",
            "change_requested",
            "authorized",
            "consent",
            "price_confirmed",
            "entitled",
        ),
        ("balance_due_cents", "subscription_day", "quoted_price_cents"),
    ),
    "healthcare": (
        (
            "staff_review",
            "cancel_requested",
            "reschedule_requested",
            "booking_requested",
            "authorized",
            "consent",
            "slot_available",
        ),
        ("appointment_day", "request_day"),
    ),
    "public-sector": (
        (
            "requires_office_review",
            "documents_complete",
            "submission_requested",
            "authorized",
            "consent",
            "appointment_requested",
            "slot_available",
            "submitted",
        ),
        ("case_day", "document_count"),
    ),
}


def admin_resolution(domain: str, row: dict[str, Any]) -> str:
    """Resolve verified facts according to the declared fictional policy order."""
    if domain == "media":
        if row["security_alert"]:
            return "escalate_security"
        if row["service_outage"]:
            return "report_outage"
        if row["cancel_requested"] or row["change_requested"]:
            if not row["authorized"]:
                return "request_authorization"
            if not row["consent"]:
                return "request_consent"
            if row["cancel_requested"]:
                return "cancel_subscription"
            return "change_subscription" if row["price_confirmed"] else "request_quote"
        if not row["entitled"]:
            return "route_access"
        return (
            "route_billing" if row["balance_due_cents"] > 0 else "report_subscription"
        )
    if domain == "healthcare":
        if row["staff_review"]:
            return "route_staff"
        if any(
            row[key]
            for key in ("cancel_requested", "reschedule_requested", "booking_requested")
        ):
            if not row["authorized"]:
                return "request_authorization"
            if not row["consent"]:
                return "request_consent"
            if row["cancel_requested"]:
                return "cancel_appointment"
            if not row["slot_available"]:
                return "offer_alternatives"
            return (
                "reschedule_appointment"
                if row["reschedule_requested"]
                else "book_appointment"
            )
        return "report_appointment"
    if domain != "public-sector":
        raise ValueError("Unknown administrative domain")
    if row["requires_office_review"]:
        return "route_office"
    if row["submission_requested"]:
        if not row["authorized"]:
            return "request_authorization"
        if not row["documents_complete"]:
            return "request_documents"
        return "submit_application" if row["consent"] else "request_consent"
    if row["appointment_requested"]:
        if not row["authorized"]:
            return "request_authorization"
        if not row["consent"]:
            return "request_consent"
        return "book_appointment" if row["slot_available"] else "offer_alternatives"
    return "report_submitted" if row["submitted"] else "report_draft"


def admin_case(
    domain: str, family: str, variant: int, rng: random.Random
) -> dict[str, Any]:
    """Build distinct fictional cases, including conflicting lower-priority facts."""
    if domain == "media":
        row: dict[str, Any] = {
            "security_alert": False,
            "service_outage": False,
            "cancel_requested": False,
            "change_requested": False,
            "authorized": True,
            "consent": True,
            "price_confirmed": True,
            "entitled": True,
            "balance_due_cents": 0,
            "subscription_day": rng.randint(1, 100000),
            "quoted_price_cents": rng.randint(0, 100000),
        }
        if family == "missing_verification":
            row.update(verified=False, security_alert=True)
        elif family == "security_incident":
            row.update(security_alert=True, service_outage=True, cancel_requested=True)
        elif family == "outage":
            row.update(service_outage=True, change_requested=True)
        elif family in {"cancellation", "subscription_change"}:
            row.update(
                cancel_requested=family == "cancellation",
                change_requested=family == "subscription_change",
                authorized=variant % 4 != 0,
                consent=variant % 4 != 1,
                price_confirmed=variant % 4 != 2,
            )
        elif family == "access":
            row.update(entitled=False, balance_due_cents=100)
        elif family == "billing":
            row["balance_due_cents"] = rng.randint(1, 100000)
        return row
    if domain == "healthcare":
        row = {
            "staff_review": False,
            "cancel_requested": False,
            "reschedule_requested": False,
            "booking_requested": True,
            "authorized": True,
            "consent": True,
            "slot_available": True,
            "appointment_day": rng.randint(1, 100000),
            "request_day": rng.randint(0, 100000),
        }
        if family == "missing_verification":
            row.update(verified=False, staff_review=True)
        elif family == "staff_review":
            row.update(staff_review=True, cancel_requested=True)
        elif family == "unauthorized_change":
            row.update(authorized=False, consent=False, slot_available=False)
        elif family == "no_consent":
            row.update(consent=False, slot_available=False)
        elif family == "no_slot":
            row["slot_available"] = False
        elif family == "reschedule":
            row.update(booking_requested=False, reschedule_requested=True)
        elif family == "cancellation":
            row.update(
                booking_requested=False, cancel_requested=True, slot_available=False
            )
        return row
    if domain != "public-sector":
        raise ValueError("Unknown administrative domain")
    row = {
        "requires_office_review": False,
        "documents_complete": True,
        "submission_requested": True,
        "authorized": True,
        "consent": True,
        "appointment_requested": False,
        "slot_available": True,
        "submitted": False,
        "case_day": rng.randint(1, 100000),
        "document_count": rng.randint(1, 10000),
    }
    if family == "missing_verification":
        row.update(verified=False, requires_office_review=True)
    elif family == "office_review":
        row["requires_office_review"] = True
    elif family == "incomplete_documents":
        row.update(documents_complete=False, consent=False)
    elif family == "unauthorized_submission":
        row.update(authorized=False, documents_complete=False, consent=False)
    elif family == "unconfirmed_submission":
        row["consent"] = False
    elif family == "appointment":
        row.update(
            submission_requested=False,
            appointment_requested=True,
            authorized=variant % 4 != 0,
            consent=variant % 4 != 1,
            slot_available=variant % 4 != 2,
        )
    elif family == "case_status":
        row.update(submission_requested=False, submitted=variant % 2 == 0)
    return row
