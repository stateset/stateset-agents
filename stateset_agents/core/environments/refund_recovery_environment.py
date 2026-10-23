"""Synthetic authorization and recovery tasks with explicit outcome knowledge.

Faults are deterministic fixtures, never intercepted real infrastructure errors.
All ledgers, authorization grants, and reconciliation fences are in memory. This
environment establishes neither real payment safety nor durable idempotency.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from stateset_agents.core.environment_base import (
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from stateset_agents.core.environments.json_action import parse_json_action
from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
)
from stateset_agents.core.refund_tools import (
    RefundOrderFacts,
    RefundResolutionRequest,
    SyntheticRefundResolver,
)
from stateset_agents.core.trajectory import ConversationTurn

RECOVERY_PROTOCOL = "refund-recovery-v1"
RECOVERY_FAMILIES = (
    "clean",
    "unauthorized",
    "lookup_transient",
    "stale_chargeback",
    "stale_refund",
    "authorization_revoked",
    "lost_ack_applied",
    "lost_ack_not_applied",
    "reconcile_unavailable",
    "reconcile_pending",
)


class _InjectedLookupUnavailable(Exception):
    pass


class _InjectedAccessDenied(Exception):
    pass


class _InjectedNotApplied(Exception):
    pass


class _InjectedAcknowledgmentLost(Exception):
    pass


class RefundRecoveryEnvironment(Environment):
    """Reward safe recovery choices over a bound fictional order.

    Every episode starts ready. Model tools express intent only; a trusted
    resolver computes amounts and policy priority. Ambiguous submissions must
    be reconciled before retry or completion. A confirmed not-applied response
    includes a synthetic fence guaranteeing that the earlier attempt cannot
    commit later. Pending/unavailable reconciliation requires a handoff.
    """

    def __init__(self) -> None:
        super().__init__(max_turns=6)
        self._state: EnvironmentState | None = None
        self._resolver: SyntheticRefundResolver | None = None
        self._events: list[dict[str, Any]] = []
        self._row: dict[str, Any] = {}
        self._fault = ""
        self._authorized = False
        self._lookup_attempts = self._submit_attempts = 0
        self._receipt: dict[str, Any] | None = None

    @staticmethod
    def validate_scenario(row: dict[str, Any]) -> None:
        """Validate task parameters without exposing the injected fault."""
        RefundPolicyEnvironment.validate_scenario(row)
        if row.get("recovery") not in RECOVERY_FAMILIES:
            raise ValueError("Unknown recovery fixture")
        if row["recovery"] == "stale_chargeback" and row["chargeback_open"]:
            raise ValueError(
                "Stale-chargeback fixture needs an initially closed chargeback"
            )
        if (
            row["recovery"] == "stale_refund"
            and row["refunded_cents"] == row["paid_cents"]
        ):
            raise ValueError("Stale-refund fixture needs a remaining balance")

    def observation_context(self, state: EnvironmentState) -> dict[str, Any]:
        """Keep fault labels, commit knowledge, and private counters hidden."""
        return {}

    def execution_events(self) -> list[dict[str, Any]]:
        """Return detached backend-attempt and effect evidence for the last intent."""
        return copy.deepcopy(self._events)

    def _new_resolver(self) -> SyntheticRefundResolver:
        return SyntheticRefundResolver(
            order_id=self._row["order_id"],
            idempotency_key=f"{RECOVERY_PROTOCOL}:{self._row['order_id']}",
            lookup=self._lookup,
            submit=self._submit,
        )

    async def reset(self, scenario: dict[str, Any] | None = None) -> EnvironmentState:
        """Start a private, isolated synthetic recovery episode."""
        self._state, self._resolver = None, None
        self._events, self._row, self._receipt = [], {}, None
        self._authorized = False
        self._lookup_attempts = self._submit_attempts = 0
        row = copy.deepcopy(scenario or {})
        self.validate_scenario(row)
        self._row, self._fault = row, row["recovery"]
        self._authorized = self._fault != "unauthorized"
        self._resolver = self._new_resolver()
        prompt = (
            "Resolve the refund request for the order bound to this session. "
            "The trusted tool chooses the policy action and exact integer cents. "
            "Use one JSON object per turn, exactly tool and args with unique keys, "
            "no prose, fences, or separate tool calls. Start with "
            '{"tool":"resolve_refund","args":{"intent":"resolve_refund"}}. '
            "Never supply an order ID, amount, reason, or facts. Follow the trusted "
            "tool status: applied -> finish resolved; unauthorized -> finish "
            "unauthorized; retryable -> retry resolve_refund. outcome_unknown "
            '-> call {"tool":"reconcile","args":{}} before any retry or finish. '
            "Reconcile status applied -> finish resolved; not_applied with "
            "retry_safe true -> retry resolve_refund; pending or unavailable "
            "-> finish handoff. Finish using exactly "
            '{"tool":"finish","args":{"outcome":"resolved"}}, '
            'or outcome "unauthorized" or "handoff" as required by the status. '
            "Never retry an uncertain submission before a trusted not-applied "
            "fence, and never guess whether a missing acknowledgment committed."
        )
        self._state = EnvironmentState(
            episode_id=row["order_id"],
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={
                "messages": [{"role": "user", "content": prompt}],
                "phase": "ready",
                "violations": 0,
                "resolution_count": 0,
                "refunded_cents": 0,
                "duplicate_attempts": 0,
                "reconciliations": 0,
            },
        )
        return self._state

    def _event(self, operation: str, result: dict[str, Any], **details: Any) -> None:
        self._events.append(
            copy.deepcopy({"operation": operation, **details, "result": result})
        )

    async def _lookup(self, order_id: str) -> dict[str, Any]:
        self._lookup_attempts += 1
        if order_id != self._row["order_id"] or not self._authorized:
            self._event("lookup_order", {"status": "unauthorized"})
            raise _InjectedAccessDenied
        if self._fault == "lookup_transient" and self._lookup_attempts == 1:
            self._event("lookup_order", {"status": "unavailable", "retry_safe": True})
            raise _InjectedLookupUnavailable
        # Facts are never sourced from model text or private evaluator annotations.
        facts = {
            name: self._row[name] for name in RefundOrderFacts.__dataclass_fields__
        }
        self._event("lookup_order", {"status": "ok", "facts": facts})
        return facts

    def _commit(self, request: RefundResolutionRequest) -> dict[str, Any]:
        assert self._state is not None
        if self._receipt is not None or self._state.context["resolution_count"]:
            raise RuntimeError("Synthetic ledger attempted a second resolution")
        action = request.action()
        self._state.context["resolution_count"] += 1
        if action["tool"] == "refund":
            self._state.context["refunded_cents"] += action["args"]["amount_cents"]
        self._receipt = {"applied": True, "action": action}
        return copy.deepcopy(self._receipt)

    async def _submit(self, request: RefundResolutionRequest) -> dict[str, Any]:
        self._submit_attempts += 1
        details = {
            "request": request.action(),
            "idempotency_key": request.idempotency_key,
        }
        if self._fault == "authorization_revoked" and self._submit_attempts == 1:
            self._authorized = False
        if not self._authorized:
            self._event(
                "submit_resolution",
                {"status": "unauthorized", "committed": False},
                **details,
            )
            raise _InjectedAccessDenied
        if self._submit_attempts == 1 and self._fault.startswith("stale_"):
            if self._fault == "stale_chargeback":
                self._row["chargeback_open"] = True
            else:
                self._row["refunded_cents"] = self._row["paid_cents"]
        if request.facts != RefundOrderFacts.from_lookup(self._row):
            self._event(
                "submit_resolution",
                {"status": "stale_facts", "committed": False, "retry_safe": True},
                **details,
            )
            raise _InjectedNotApplied
        if self._receipt is not None:
            self._event(
                "submit_resolution",
                {
                    "status": "applied",
                    "deduplicated": True,
                    "committed": False,
                    "receipt": self._receipt,
                },
                **details,
            )
            return copy.deepcopy(self._receipt)
        if self._submit_attempts == 1 and self._fault in {
            "lost_ack_applied",
            "lost_ack_not_applied",
            "reconcile_unavailable",
            "reconcile_pending",
        }:
            committed = self._fault != "lost_ack_not_applied"
            if committed:
                self._commit(request)
            # 'committed' is evaluator evidence; the model sees only outcome_unknown.
            self._event(
                "submit_resolution",
                {"status": "outcome_unknown", "committed": committed},
                **details,
            )
            raise _InjectedAcknowledgmentLost
        receipt = self._commit(request)
        self._event(
            "submit_resolution",
            {"status": "applied", "committed": True, "receipt": receipt},
            **details,
        )
        return receipt

    def _reconcile(self) -> dict[str, Any]:
        assert self._state is not None
        ctx = self._state.context
        ctx["reconciliations"] += 1
        if self._fault in {"reconcile_unavailable", "reconcile_pending"}:
            status = (
                "unavailable" if self._fault == "reconcile_unavailable" else "pending"
            )
            ctx["phase"] = "handoff"
            result = {"status": status, "retry_safe": False}
        elif self._receipt is not None:
            ctx["phase"] = "applied"
            result = {"status": "applied", "receipt": copy.deepcopy(self._receipt)}
        else:
            # In this synchronous fixture the failed attempt cannot commit later.
            # A real service must fence pending work before returning this result.
            ctx["phase"] = "retryable"
            self._resolver = self._new_resolver()
            result = {"status": "not_applied", "retry_safe": True, "fenced": True}
        self._event("reconcile", result)
        return result

    async def _resolve(self) -> dict[str, Any]:
        assert self._state is not None and self._resolver is not None
        ctx = self._state.context
        try:
            result = await self._resolver.resolve_refund(intent="resolve_refund")
        except _InjectedAccessDenied:
            ctx["phase"] = "unauthorized"
            return {"status": "unauthorized", "retry_safe": False}
        except (_InjectedLookupUnavailable, _InjectedNotApplied):
            ctx["phase"] = "retryable"
            self._resolver = self._new_resolver()
            return {"status": "retryable", "retry_safe": True}
        except _InjectedAcknowledgmentLost:
            ctx["phase"] = "unknown"
            return {"status": "outcome_unknown", "retry_safe": False}
        ctx["phase"] = "applied"
        return {"status": "applied", "receipt": result["receipt"]}

    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> tuple[EnvironmentState, float, bool, dict[str, Any]]:
        """Score recovery choices; only explicitly injected faults become observations."""
        if state is not self._state or state.is_done:
            raise ValueError("Reset required before stepping the current episode")
        state.turn_count += 1
        self._events = []
        ctx, finish, success = state.context, False, False
        try:
            tool, args = parse_json_action(action)
            if tool == "resolve_refund":
                if args != {"intent": "resolve_refund"}:
                    raise ValueError("Only a bound resolution intent is accepted")
                if ctx["phase"] not in {"ready", "retryable"}:
                    ctx["duplicate_attempts"] += 1
                    raise ValueError("Resolution retry is not authorized in this phase")
            elif tool == "reconcile":
                if args or ctx["phase"] != "unknown":
                    raise ValueError(
                        "Reconcile requires an uncertain submission and no args"
                    )
            elif tool == "finish":
                # A non-string outcome (an unhashable list) is an invalid
                # action, never a TypeError out of the set lookup.
                if (
                    set(args) != {"outcome"}
                    or not isinstance(args["outcome"], str)
                    or args["outcome"] not in {"resolved", "unauthorized", "handoff"}
                ):
                    raise ValueError("Finish requires an explicit valid outcome")
                finish = True
                expected = {
                    "applied": "resolved",
                    "unauthorized": "unauthorized",
                    "handoff": "handoff",
                }.get(ctx["phase"])
                if args["outcome"] != expected:
                    raise ValueError(
                        "Finish outcome is not supported by backend knowledge"
                    )
            else:
                raise ValueError("Unknown recovery tool")
        except ValueError as exc:
            ctx["violations"] += 1
            observation: dict[str, Any] = {"error": str(exc)}
        else:
            # Real/unplanned exceptions are outside the model-error catch block.
            if tool == "resolve_refund":
                observation = await self._resolve()
            elif tool == "reconcile":
                observation = self._reconcile()
            else:
                observation = {"finished": True, "outcome": args["outcome"]}
                success = ctx["violations"] == 0
        done = finish or state.turn_count >= self.max_turns
        if done:
            state.status = EpisodeStatus.COMPLETED if finish else EpisodeStatus.TIMEOUT
        return (
            state,
            ((1.0 if success else -1.0) if done else 0.0),
            done,
            {
                "messages": [
                    {"role": "user", "content": json.dumps(observation, sort_keys=True)}
                ],
                "metrics": {
                    "task_success": float(success),
                    "policy_violations": ctx["violations"],
                    "tool_calls": state.turn_count,
                    "backend_calls": self._lookup_attempts
                    + self._submit_attempts
                    + ctx["reconciliations"],
                    "resolution_count": ctx["resolution_count"],
                    "refunded_cents": ctx["refunded_cents"],
                    "duplicate_attempts": ctx["duplicate_attempts"],
                    "reconciliations": ctx["reconciliations"],
                },
            },
        )
