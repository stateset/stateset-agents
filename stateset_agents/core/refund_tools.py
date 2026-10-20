"""Explicit delegation tools for the synthetic refund-policy fixture.

This module implements a fictional policy, not a merchant payment integration.
Authorization, durable idempotency, and atomic fact revalidation belong to the
trusted backend. Model output supplies intent only; it never supplies order facts.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .function_calling import FunctionDefinition


@dataclass(frozen=True)
class RefundOrderFacts:
    """Immutable, policy-relevant snapshot returned by a trusted lookup."""

    order_id: str
    paid_cents: int
    refunded_cents: int
    days_since_delivery: int
    return_window_days: int
    status: str
    chargeback_open: bool

    def __post_init__(self) -> None:
        if not isinstance(self.order_id, str) or not self.order_id.strip():
            raise ValueError("order_id must be nonempty text")
        for name in (
            "paid_cents",
            "refunded_cents",
            "days_since_delivery",
            "return_window_days",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be nonnegative integer cents or days")
        if self.paid_cents < 1 or self.refunded_cents > self.paid_cents:
            raise ValueError("Refunded amount must not exceed positive payment")
        if self.status not in ("delivered", "processing", "cancelled"):
            raise ValueError("Unknown order status")
        if type(self.chargeback_open) is not bool:
            raise ValueError("chargeback_open must be boolean")

    @classmethod
    def from_lookup(cls, row: Mapping[str, Any]) -> RefundOrderFacts:
        """Read facts only; ignore notes, answer labels, and other metadata."""
        fields = cls.__dataclass_fields__
        if not isinstance(row, Mapping) or any(name not in row for name in fields):
            raise ValueError("Trusted lookup must contain every policy fact")
        return cls(**{name: row[name] for name in fields})


@dataclass(frozen=True)
class RefundResolutionRequest:
    """Backend submission binding an authorized snapshot to a stable key.

    The backend must atomically verify these facts are still current and apply
    at most one resolution per authorized order and idempotency key. A stale
    snapshot must be rejected, including changes to chargeback or prior refunds.
    Acknowledgment receipts must include ``applied: True``; rejected or uncertain
    responses must never be acknowledged as an applied resolution.
    """

    facts: RefundOrderFacts
    idempotency_key: str
    policy: str = "synthetic-refund-policy-v2"

    def action(self) -> dict[str, Any]:
        """Compute the exact action in policy priority order using integer cents."""
        facts = self.facts
        args: dict[str, Any] = {"order_id": facts.order_id}
        if facts.chargeback_open:
            tool, reason = "escalate", "chargeback"
        elif facts.refunded_cents == facts.paid_cents:
            tool, reason = "deny", "already_refunded"
        elif facts.status != "delivered":
            tool, reason = "deny", "not_delivered"
        elif facts.days_since_delivery > facts.return_window_days:
            tool, reason = "deny", "outside_window"
        else:
            args["amount_cents"] = facts.paid_cents - facts.refunded_cents
            return {"tool": "refund", "args": args}
        args["reason"] = reason
        return {"tool": tool, "args": args}


class SyntheticRefundResolver:
    """Bind one authorized order to a model-callable resolution intent.

    Use one instance per authorized order on one event loop. Concurrent calls
    share one lookup/submission and successful receipts are copied on return.
    Once submission starts, any exception or cancellation leaves the outcome
    uncertain: this instance refuses another submission until the caller
    reconciles with the backend. No automatic retry or claim of exactly-once
    execution across processes is made.
    """

    def __init__(
        self,
        *,
        order_id: str,
        idempotency_key: str,
        lookup: Callable[[str], Awaitable[Mapping[str, Any]]],
        submit: Callable[[RefundResolutionRequest], Awaitable[dict[str, Any]]],
    ) -> None:
        for name, value in (
            ("order_id", order_id),
            ("idempotency_key", idempotency_key),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty text")
        self._order_id = order_id
        self._key = idempotency_key
        self._lookup = lookup
        self._submit = submit
        self._lock = asyncio.Lock()
        self._uncertain = False
        self._result: dict[str, Any] | None = None

    async def resolve_refund(self, *, intent: str) -> dict[str, Any]:
        """Explicitly delegate resolution; accept no model-selected facts/actions."""
        if type(intent) is not str or intent != "resolve_refund":
            raise ValueError("Explicit resolve_refund intent is required")
        async with self._lock:
            if self._result is not None:
                return copy.deepcopy(self._result)
            if self._uncertain:
                raise RuntimeError(
                    "Resolution outcome uncertain; reconcile with backend"
                )
            facts = RefundOrderFacts.from_lookup(await self._lookup(self._order_id))
            if facts.order_id != self._order_id:
                raise ValueError("Trusted lookup returned a different order")
            request = RefundResolutionRequest(facts, self._key)
            # Set before awaiting: cancellation may follow a committed side effect.
            self._uncertain = True
            receipt = await self._submit(request)
            if not isinstance(receipt, dict) or receipt.get("applied") is not True:
                raise ValueError(
                    "Backend must explicitly acknowledge applied resolution"
                )
            result = copy.deepcopy(
                {
                    "policy": request.policy,
                    "action": request.action(),
                    "receipt": receipt,
                }
            )
            self._result = result
            self._uncertain = False
            return copy.deepcopy(result)

    def as_function(self) -> FunctionDefinition:
        """Register only the bound intent handler with FunctionCallingMixin."""
        return FunctionDefinition(
            name="resolve_refund",
            description=(
                "Resolve this authorized order under synthetic-refund-policy-v2. "
                "The trusted tool looks up facts and selects refund, denial, or "
                "escalation. Never supply order IDs, amounts, reasons, or facts."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "enum": ["resolve_refund"]}
                },
                "required": ["intent"],
                "additionalProperties": False,
            },
            handler=self.resolve_refund,
            strict=True,
        )
