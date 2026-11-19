"""Durable intent delegation with backend authorization and reconciliation."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from stateset_agents.core.function_calling import FunctionDefinition
from stateset_agents.core.refund_tools import RefundOrderFacts
from stateset_agents.evaluation.agent_runs import content_hash

from .storage import private_directory, read_json, workspace_lock, write_json


class RefundBackend(Protocol):
    """Trusted merchant adapter; the model supplies none of these parameters.

    ``commit`` must atomically recheck authorization, policy version and facts,
    and enforce idempotency durably. ``reconcile`` must query that same operation
    identity. A local journal alone cannot guarantee exactly-once payments.
    """

    policy_version: str

    async def authorize(self, principal: str, order_id: str) -> None:
        """Raise unless this principal may resolve this order now."""
        ...

    async def lookup(self, order_id: str) -> RefundOrderFacts:
        """Return current integer-cent facts from the trusted system of record."""
        ...

    def decide(self, facts: RefundOrderFacts) -> dict[str, Any]:
        """Apply the merchant's versioned deterministic policy."""
        ...

    async def commit(self, request: dict[str, Any]) -> dict[str, Any]:
        """Atomically revalidate the complete bound request and apply at most once."""
        ...

    async def reconcile(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Return a confirmed receipt, or None while the outcome remains uncertain."""
        ...


class DurableRefundResolver:
    """One authorized operation, serialized across processes and reconciled on loss.

    Only an explicit model intent is accepted. Unknown commit outcomes are never
    automatically resubmitted, even after process restart. A policy change requires
    reconciliation of an existing attempt, not a new key for the same payment.
    """

    def __init__(
        self,
        root: Path,
        *,
        principal: str,
        order_id: str,
        idempotency_key: str,
        backend: RefundBackend,
    ) -> None:
        for value in (principal, order_id, idempotency_key, backend.policy_version):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "Principal, order, operation key and policy must be nonempty"
                )
        self.identity = {
            "principal": principal,
            "order_id": order_id,
            "idempotency_key": idempotency_key,
        }
        self.root = private_directory(
            private_directory(root) / content_hash(self.identity)
        )
        self.backend = backend
        self._lock = asyncio.Lock()

    @staticmethod
    def _receipt(request: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
        if (
            not isinstance(receipt, dict)
            or receipt.get("applied") is not True
            or receipt.get("request_sha256") != content_hash(request)
            or not isinstance(receipt.get("transaction_id"), str)
            or not receipt["transaction_id"]
        ):
            raise ValueError(
                "Backend receipt must confirm the exact request and transaction"
            )
        return copy.deepcopy(receipt)

    async def resolve_refund(self, *, intent: str) -> dict[str, Any]:
        """Authorize, compute and submit once, or return the confirmed receipt."""
        if intent != "resolve_refund":
            raise ValueError("Explicit resolve_refund intent is required")
        async with self._lock:
            with workspace_lock(self.root):
                await self.backend.authorize(
                    self.identity["principal"], self.identity["order_id"]
                )
                attempt = self.root / "attempt.json"
                if attempt.exists():
                    request = read_json(attempt)
                    self._identity(request)
                    if (self.root / "receipt.json").exists():
                        return self._receipt(
                            request, read_json(self.root / "receipt.json")
                        )
                    raise ValueError(
                        "Outcome uncertain; reconcile with the trusted backend"
                    )
                facts = await self.backend.lookup(self.identity["order_id"])
                if (
                    not isinstance(facts, RefundOrderFacts)
                    or facts.order_id != self.identity["order_id"]
                ):
                    raise ValueError(
                        "Lookup returned a different order or invalid facts"
                    )
                action = self.backend.decide(facts)
                if not isinstance(action, dict) or not action:
                    raise ValueError("Trusted policy must return an action")
                request = {
                    **self.identity,
                    "facts": asdict(facts),
                    "action": action,
                    "policy_version": self.backend.policy_version,
                }
                # Seal before awaiting a side effect: cancellation can follow commit.
                write_json(attempt, request)
                receipt = self._receipt(
                    request, await self.backend.commit(copy.deepcopy(request))
                )
                write_json(self.root / "receipt.json", receipt)
                return receipt

    def _identity(self, request: dict[str, Any]) -> None:
        if any(request.get(key) != value for key, value in self.identity.items()):
            raise ValueError(
                "Retained operation belongs to a different principal/order/key"
            )

    async def reconcile(self) -> dict[str, Any]:
        """Query an existing operation without issuing another payment request."""
        async with self._lock:
            with workspace_lock(self.root):
                await self.backend.authorize(
                    self.identity["principal"], self.identity["order_id"]
                )
                request = read_json(self.root / "attempt.json")
                self._identity(request)
                if (self.root / "receipt.json").exists():
                    return self._receipt(request, read_json(self.root / "receipt.json"))
                result = await self.backend.reconcile(copy.deepcopy(request))
                if result is None:
                    raise ValueError(
                        "Backend has not confirmed the operation; do not resubmit"
                    )
                receipt = self._receipt(request, result)
                write_json(self.root / "receipt.json", receipt)
                return receipt

    def as_function(self) -> FunctionDefinition:
        """Expose intent only; bind authorization, policy and money in trusted code."""
        return FunctionDefinition(
            name="resolve_refund",
            description="Resolve this authorized order through the trusted policy and payment backend.",
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
