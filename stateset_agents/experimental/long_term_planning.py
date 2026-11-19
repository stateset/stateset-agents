"""
Long-term planning utilities for conversational agents.

Provides a lightweight planning manager that can generate and maintain
multi-step plans, then inject them into the prompt as system context.
"""

from __future__ import annotations

import asyncio
import copy
import re
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any
from uuid import uuid4

from stateset_agents.core.multiturn_context import ConversationConflict

_PLANNING_GUARD: ContextVar[Callable[[], AbstractContextManager[Any]] | None] = (
    ContextVar("planning_commit_guard", default=None)
)


@contextmanager
def _planning_scope(guard: Callable[[], AbstractContextManager[Any]]) -> Iterator[None]:
    """Bind host-operation guards without serializing callbacks or changing APIs."""
    token = _PLANNING_GUARD.set(guard)
    try:
        yield
    finally:
        _PLANNING_GUARD.reset(token)


class PlanStatus(str, Enum):
    """Status for individual plan steps."""

    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    BLOCKED = "blocked"


@dataclass
class PlanStep:
    """Single step in a plan."""

    description: str
    status: PlanStatus = PlanStatus.PENDING
    step_id: str = ""
    notes: str | None = None
    updated_at: datetime = field(default_factory=datetime.now)

    def mark(self, status: PlanStatus, notes: str | None = None) -> None:
        self.status = status
        if notes:
            self.notes = notes
        self.updated_at = datetime.now()


@dataclass
class Plan:
    """Plan with ordered steps for a goal."""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def ensure_active(self) -> None:
        for step in self.steps:
            if step.status == PlanStatus.ACTIVE:
                return
        for step in self.steps:
            if step.status == PlanStatus.PENDING:
                step.status = PlanStatus.ACTIVE
                return

    def advance(self) -> None:
        """Mark active step done and advance to the next pending step."""
        for step in self.steps:
            if step.status == PlanStatus.ACTIVE:
                step.mark(PlanStatus.DONE)
                break
        for step in self.steps:
            if step.status == PlanStatus.PENDING:
                step.mark(PlanStatus.ACTIVE)
                break
        self.updated_at = datetime.now()

    def update_step(
        self,
        step_id: str | None = None,
        step_index: int | None = None,
        status: PlanStatus | None = None,
        notes: str | None = None,
    ) -> None:
        step = None
        if step_id:
            for candidate in self.steps:
                if candidate.step_id == step_id:
                    step = candidate
                    break
        elif step_index is not None and 0 <= step_index < len(self.steps):
            step = self.steps[step_index]

        if step is None:
            return

        if status is not None:
            step.mark(status, notes=notes)
        elif notes:
            step.notes = notes
        self.updated_at = datetime.now()

    def progress(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for step in self.steps if step.status == PlanStatus.DONE)
        return float(done) / float(len(self.steps))

    def summarize(
        self,
        max_steps: int = 6,
        keep_completed: int = 2,
        include_goal: bool = True,
    ) -> str:
        lines = []
        if include_goal:
            lines.append(f"Goal: {self.goal}")

        completed = [s for s in self.steps if s.status == PlanStatus.DONE]
        active_or_pending = [s for s in self.steps if s.status != PlanStatus.DONE]

        display_steps: list[PlanStep] = []
        if keep_completed > 0:
            display_steps.extend(completed[-keep_completed:])
        display_steps.extend(active_or_pending)
        display_steps = display_steps[:max_steps]

        if display_steps:
            lines.append("Steps:")
            for idx, step in enumerate(display_steps, 1):
                status_tag = {
                    PlanStatus.DONE: "[x]",
                    PlanStatus.ACTIVE: "[>]",
                    PlanStatus.BLOCKED: "[!]",
                    PlanStatus.PENDING: "[ ]",
                }.get(step.status, "[ ]")
                lines.append(f"{idx}. {status_tag} {step.description}")

        return "\n".join(lines).strip()


@dataclass
class PlanningConfig:
    """Configuration for long-term planning."""

    enabled: bool = False
    max_steps: int = 6
    update_interval: int = 1
    include_plan_in_context: bool = True
    allow_goal_update: bool = True
    goal_min_chars: int = 8
    keep_completed: int = 2
    plan_prefix: str = "Long-term plan"


class Planner:
    """Planner interface for generating and revising plans."""

    def create_plan(self, goal: str, config: PlanningConfig) -> Plan:
        raise NotImplementedError

    def revise_plan(self, plan: Plan, goal: str, config: PlanningConfig) -> Plan:
        return self.create_plan(goal, config)


class HeuristicPlanner:
    """Simple deterministic planner that splits goals into steps."""

    def create_plan(self, goal: str, config: PlanningConfig) -> Plan:
        steps = _split_goal_into_steps(goal, config.max_steps)
        plan_steps = [
            PlanStep(description=step, step_id=f"step_{idx + 1}")
            for idx, step in enumerate(steps)
        ]
        plan = Plan(goal=goal, steps=plan_steps)
        plan.ensure_active()
        return plan

    def revise_plan(self, plan: Plan, goal: str, config: PlanningConfig) -> Plan:
        if goal != plan.goal:
            return self.create_plan(goal, config)
        return plan


class PlanningManager:
    """Manages long-term plans across conversations."""

    def __init__(
        self,
        config: PlanningConfig | None = None,
        planner: Any | None = None,
        goal_extractor: (
            Callable[[list[dict[str, str]], dict[str, Any]], str | None] | None
        ) = None,
    ):
        self.config = config or PlanningConfig()
        self.planner = planner or HeuristicPlanner()
        self.goal_extractor = goal_extractor or _default_goal_extractor
        self._plans: dict[str, Plan] = {}
        self._turn_counters: dict[str, int] = {}
        self._last_updates: dict[str, int] = {}
        self._lock = RLock()
        self._revisions: dict[str, int] = {}
        self._operations: dict[str, dict[str, Any]] = {}
        self._last_operations: dict[str, dict[str, Any]] = {}

    def __getstate__(self) -> dict[str, Any]:
        with self._lock:
            if self._operations:
                raise RuntimeError("Cannot copy a planning manager during an operation")
            return {key: value for key, value in vars(self).items() if key != "_lock"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = RLock()
        for name in ("_revisions", "_operations", "_last_operations"):
            self.__dict__.setdefault(name, {})

    def get_plan(self, conversation_id: str) -> Plan | None:
        """Return the current plan; concurrent direct mutations are unsupported."""
        with self._lock:
            return self._plans.get(conversation_id)

    def get_status(self, conversation_id: str) -> dict[str, Any]:
        """Inspect planning progress without executing a planner or consuming intent."""
        with self._lock:
            return copy.deepcopy(
                {
                    "revision": self._revisions.get(conversation_id, 0),
                    "has_plan": conversation_id in self._plans,
                    "turn_count": self._turn_counters.get(conversation_id, 0),
                    "last_update": self._last_updates.get(conversation_id, 0),
                    "active": self._operations.get(conversation_id),
                    "last_finished": self._last_operations.get(conversation_id),
                }
            )

    def clear_conversation(self, conversation_id: str) -> None:
        """Remove planning state for a conversation."""
        with self._lock:
            self._plans.pop(conversation_id, None)
            self._turn_counters.pop(conversation_id, None)
            self._last_updates.pop(conversation_id, None)
            self._last_operations.pop(conversation_id, None)
            if conversation_id in self._operations:
                self._revisions[conversation_id] = (
                    self._revisions.get(conversation_id, 0) + 1
                )
            else:
                self._revisions.pop(conversation_id, None)

    def build_plan_message(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, str] | None:
        """Prepare a plan off-lock and commit only to the revision that admitted it."""
        if not self.config.enabled or not self.config.include_plan_in_context:
            return None

        context = copy.deepcopy(context or {})
        messages = copy.deepcopy(messages)
        conversation_id = str(context.get("conversation_id") or "default")
        guard = _PLANNING_GUARD.get() or nullcontext
        with guard(), self._lock:
            if conversation_id in self._operations:
                raise ConversationConflict(
                    "Another planning operation owns this conversation"
                )
            original = self._plans.get(conversation_id)
            baseline = copy.deepcopy(original)
            candidate = copy.deepcopy(original)
            config = copy.deepcopy(self.config)
            planner, extractor = self.planner, self.goal_extractor
            revision = self._revisions.get(conversation_id, 0)
            turn_count = self._turn_counters.get(conversation_id, 0) + 1
            last_update = self._last_updates.get(conversation_id, 0)
            receipt = {
                "id": uuid4().hex,
                "state": "running",
                "phase": "preparing",
                "callback_status": "not_started",
                "error_type": None,
            }
            self._operations[conversation_id] = receipt
        try:
            explicit_goal = _extract_explicit_goal(context)
            update_needed = (
                candidate is None
                or turn_count - last_update >= config.update_interval
                or isinstance(context.get("plan_update"), dict)
                or bool(explicit_goal)
            )
            if update_needed:
                update = context.get("plan_update")
                if candidate is not None and isinstance(update, dict):
                    _apply_plan_update(candidate, update)
                goal = explicit_goal
                if goal is None and candidate is None:
                    with guard(), self._lock:
                        self._check_revision(conversation_id, revision, receipt)
                        receipt.update(phase="planning", callback_status="unconfirmed")
                    goal = extractor(messages, context)
                    with self._lock:
                        receipt["callback_status"] = "returned"
                if goal and len(goal) >= config.goal_min_chars:
                    if candidate is None or (
                        config.allow_goal_update and explicit_goal
                    ):
                        with guard(), self._lock:
                            self._check_revision(conversation_id, revision, receipt)
                            receipt.update(
                                phase="planning", callback_status="unconfirmed"
                            )
                        candidate = (
                            planner.create_plan(goal, config)
                            if candidate is None
                            else planner.revise_plan(candidate, goal, config)
                        )
                        with self._lock:
                            receipt["callback_status"] = "returned"
                last_update = turn_count
            if candidate is not None and not isinstance(candidate, Plan):
                raise TypeError("Planner must return a Plan")
            prepared = copy.deepcopy(candidate)
            with guard(), self._lock:
                self._check_revision(conversation_id, revision, receipt)
            summary = (
                prepared.summarize(
                    max_steps=config.max_steps, keep_completed=config.keep_completed
                )
                if prepared is not None
                else ""
            )
            with guard(), self._lock:
                receipt["phase"] = "committing"
                self._check_revision(conversation_id, revision, receipt)
                if (
                    self._plans.get(conversation_id) is not original
                    or original != baseline
                ):
                    raise ConversationConflict("Plan changed during preparation")
                if prepared is not None:
                    self._plans[conversation_id] = prepared
                self._turn_counters[conversation_id] = turn_count
                self._last_updates[conversation_id] = last_update
                self._revisions[conversation_id] = revision + 1
                receipt.update(state="completed", phase="finished")
            return (
                {"role": "system", "content": f"{config.plan_prefix}:\n{summary}"}
                if summary
                else None
            )
        except BaseException as error:
            with self._lock:
                receipt.update(
                    state=(
                        "cancelled"
                        if isinstance(error, asyncio.CancelledError)
                        else "failed"
                    ),
                    error_type=type(error).__name__,
                )
            raise
        finally:
            with self._lock:
                self._operations.pop(conversation_id)
                self._last_operations[conversation_id] = copy.deepcopy(receipt)

    def _check_revision(
        self, conversation_id: str, revision: int, receipt: dict[str, Any]
    ) -> None:
        if (
            self._revisions.get(conversation_id, 0) != revision
            or self._operations.get(conversation_id) is not receipt
        ):
            raise ConversationConflict("Planning state changed during preparation")

    def apply_update(
        self,
        conversation_id: str,
        update: dict[str, Any],
    ) -> None:
        """Apply a detached update atomically and invalidate stale preparations."""
        with self._lock:
            plan = self._plans.get(conversation_id)
            if plan is None:
                return
            candidate = copy.deepcopy(plan)
            _apply_plan_update(candidate, copy.deepcopy(update))
            self._plans[conversation_id] = candidate
            self._revisions[conversation_id] = (
                self._revisions.get(conversation_id, 0) + 1
            )


def _apply_plan_update(plan: Plan, update: dict[str, Any]) -> None:
    action = str(update.get("action") or "").lower()
    if action == "advance":
        plan.advance()
        return
    status = update.get("status")
    if isinstance(status, str):
        try:
            status_value = PlanStatus(status)
        except ValueError:
            status_value = None
    else:
        status_value = status
    step_index = update.get("step_index")
    if isinstance(step_index, int) and step_index > 0:
        step_index -= 1
    plan.update_step(
        step_id=update.get("step_id"),
        step_index=step_index,
        status=status_value if isinstance(status_value, PlanStatus) else None,
        notes=update.get("notes"),
    )


def _default_goal_extractor(
    messages: list[dict[str, str]],
    context: dict[str, Any],
) -> str | None:
    explicit = _extract_explicit_goal(context)
    if explicit:
        return explicit

    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = str(msg.get("content") or "").strip()
            if content:
                return content
    return None


def _extract_explicit_goal(context: dict[str, Any]) -> str | None:
    explicit = context.get("plan_goal")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    explicit = context.get("goal")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    user_goal = context.get("user_goal")
    if isinstance(user_goal, str) and user_goal.strip():
        return user_goal.strip()
    scenario = context.get("scenario")
    if isinstance(scenario, dict):
        for key in ("goal", "user_goal", "context"):
            value = scenario.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _split_goal_into_steps(goal: str, max_steps: int) -> list[str]:
    cleaned = goal.strip()
    if not cleaned:
        return []

    parts = re.split(
        r"\.\s+|;|,|\band\b|\bthen\b|\bnext\b", cleaned, flags=re.IGNORECASE
    )
    steps = [part.strip() for part in parts if part.strip()]

    if not steps:
        steps = [cleaned]

    return steps[:max_steps]
